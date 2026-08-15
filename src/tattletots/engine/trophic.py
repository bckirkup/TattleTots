"""Trophic attachment: agents choose inputs that maximize metabolic yield."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.identity import stable_id_digest
from tattletots.models.stream import Stream, StreamType


def stream_attachment_key(stream: Stream) -> str:
    """Return the stable genome key for a stream attachment."""
    return stream.label or stream.id


def compute_stream_attractiveness(
    agent: Agent,
    stream: Stream,
    rng: np.random.Generator,
    *,
    grounded_multiplier: float = 1.0,
) -> float:
    """Score how attractive a stream is to an agent based on genome preferences.

    Attractiveness = preference weight * stream structured variance, optionally
    scaled by `grounded_multiplier` for grounded raw streams. The multiplier is
    domain-neutral: it only reads `Stream.stream_type`.
    """
    base_attractiveness = stream.structured_variance
    if stream.stream_type == StreamType.RAW:
        base_attractiveness *= grounded_multiplier

    # Use state override if available, else genome default
    pref = agent.state.input_preference_override
    if pref.size == 0:
        pref = agent.genome.input_preference
    if pref.size > 0:
        idx = stable_id_digest(stream_attachment_key(stream)) % len(pref)
        weight = float(pref[idx])
    else:
        weight = 1.0 + rng.normal(0, 0.1)

    return base_attractiveness * max(weight, 0.0)


def _weighted_draw(
    scores: NDArray[np.float64],
    n_select: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Draw `n_select` distinct indices, weighted by score where possible."""
    if scores.sum() <= 0:
        # Random selection if no clear preference
        return rng.choice(len(scores), size=n_select, replace=False)
    probs = scores / scores.sum()
    # Ensure enough non-zero entries for selection
    if int(np.count_nonzero(probs)) < n_select:
        return rng.choice(len(scores), size=n_select, replace=False)
    return rng.choice(len(scores), size=n_select, replace=False, p=probs)


def _score_streams(
    agent: Agent,
    streams: list[Stream],
    rng: np.random.Generator,
    grounded_multiplier: float,
) -> NDArray[np.float64]:
    return np.array(
        [
            compute_stream_attractiveness(agent, s, rng, grounded_multiplier=grounded_multiplier)
            for s in streams
        ],
        dtype=np.float64,
    )


def _reserved_grounded_count(
    n_select: int,
    n_grounded: int,
    grounded_fraction: float,
) -> int:
    """Number of input slots guaranteed to grounded raw streams."""
    if grounded_fraction <= 0.0 or n_grounded == 0:
        return 0
    requested = int(np.ceil(grounded_fraction * n_select))
    return min(requested, n_grounded, n_select)


def select_input_streams(
    agent: Agent,
    available_streams: list[Stream],
    max_inputs: int,
    rng: np.random.Generator,
    *,
    grounded_fraction: float = 0.0,
    grounded_multiplier: float = 1.0,
) -> list[str]:
    """Agent selects which streams to consume based on attractiveness.

    Self-organization: agents freely choose from any available stream
    (raw or residual). Trophic hierarchy emerges from these choices.

    `grounded_fraction` reserves that share of the input slots for grounded raw
    streams; `grounded_multiplier` scales raw-stream attractiveness in the
    unreserved competition. At the defaults (`0.0`, `1.0`) both the selection
    and the random-number consumption are identical to unreserved attachment.
    """
    if not available_streams:
        return []

    # Don't consume own output
    candidates = [s for s in available_streams if s.id != agent.state.output_stream_id]
    if not candidates:
        return []

    n_select = min(max_inputs, len(candidates))
    grounded = [s for s in candidates if s.stream_type == StreamType.RAW]
    n_reserved = _reserved_grounded_count(n_select, len(grounded), grounded_fraction)

    selected: list[str] = []
    if n_reserved > 0:
        grounded_scores = _score_streams(agent, grounded, rng, grounded_multiplier)
        reserved_indices = _weighted_draw(grounded_scores, n_reserved, rng)
        selected = [grounded[i].id for i in reserved_indices]
        remaining = [s for s in candidates if s.id not in set(selected)]
        n_select -= n_reserved
        if n_select == 0 or not remaining:
            return selected
        candidates = remaining

    scores = _score_streams(agent, candidates, rng, grounded_multiplier)
    indices = _weighted_draw(scores, min(n_select, len(candidates)), rng)
    return selected + [candidates[i].id for i in indices]


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
