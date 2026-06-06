"""Trophic attachment: agents choose inputs that maximize metabolic yield."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.stream import Stream


def compute_stream_attractiveness(agent: Agent, stream: Stream, rng: np.random.Generator) -> float:
    """Score how attractive a stream is to an agent based on genome preferences.

    Attractiveness = preference weight * stream structured variance.
    Higher structured variance = more extractable information.
    """
    base_attractiveness = stream.structured_variance

    # Use state override if available, else genome default
    pref = agent.state.input_preference_override
    if pref.size == 0:
        pref = agent.genome.input_preference
    if pref.size > 0:
        # Use a hash of stream id to get a stable index
        idx = hash(stream.id) % len(pref)
        weight = float(pref[idx])
    else:
        weight = 1.0 + rng.normal(0, 0.1)

    return base_attractiveness * max(weight, 0.0)


def select_input_streams(
    agent: Agent,
    available_streams: list[Stream],
    max_inputs: int,
    rng: np.random.Generator,
) -> list[str]:
    """Agent selects which streams to consume based on attractiveness.

    Self-organization: agents freely choose from any available stream
    (raw or residual). Trophic hierarchy emerges from these choices.
    """
    if not available_streams:
        return []

    # Don't consume own output
    candidates = [s for s in available_streams if s.id != agent.state.output_stream_id]
    if not candidates:
        return []

    scores: NDArray[np.float64] = np.array(
        [compute_stream_attractiveness(agent, s, rng) for s in candidates],
        dtype=np.float64,
    )

    # Softmax selection (stochastic to allow exploration)
    n_select = min(max_inputs, len(candidates))
    if scores.sum() <= 0:
        # Random selection if no clear preference
        indices = rng.choice(len(candidates), size=n_select, replace=False)
    else:
        # Weighted selection
        probs = scores / scores.sum()
        # Ensure enough non-zero entries for selection
        non_zero_count = int(np.count_nonzero(probs))
        if non_zero_count < n_select:
            indices = rng.choice(len(candidates), size=n_select, replace=False)
        else:
            indices = rng.choice(len(candidates), size=n_select, replace=False, p=probs)

    return [candidates[i].id for i in indices]


def compute_trophic_level(
    agent_id: str,
    agent_inputs: dict[str, list[str]],
    stream_sources: dict[str, str | None],
    memo: dict[str, float] | None = None,
) -> float:
    """Compute the trophic level of an agent (measured, not assigned).

    Trophic level = 1 + mean trophic level of input sources.
    Raw streams have level 0.
    """
    if memo is None:
        memo = {}
    if agent_id in memo:
        return memo[agent_id]

    # Prevent infinite recursion
    memo[agent_id] = 1.0

    input_ids = agent_inputs.get(agent_id, [])
    if not input_ids:
        memo[agent_id] = 1.0
        return 1.0

    source_levels: list[float] = []
    for stream_id in input_ids:
        source = stream_sources.get(stream_id)
        if source is None:
            # Raw stream: level 0
            source_levels.append(0.0)
        else:
            # Residual from another agent
            level = compute_trophic_level(source, agent_inputs, stream_sources, memo)
            source_levels.append(level)

    result = 1.0 + (sum(source_levels) / len(source_levels) if source_levels else 0.0)
    memo[agent_id] = result
    return result
