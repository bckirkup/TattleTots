"""Whistleblowing: agents that detect dishonesty in other agents' outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.stream import Stream, StreamType


def identify_whistleblower_targets(
    agent: Agent,
    streams: dict[str, Stream],
) -> list[str]:
    """Find output streams (not residuals) that a whistleblower agent consumes.

    A whistleblower consumes another agent's OUTPUT as its input,
    and detects inconsistencies between claims and reality.
    """
    targets: list[str] = []
    for stream_id in agent.state.input_stream_ids:
        stream = streams.get(stream_id)
        if stream is not None and stream.stream_type == StreamType.OUTPUT:
            targets.append(stream_id)
    return targets


def compute_dishonesty_score(
    claimed_output: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
    other_agents_consensus: NDArray[np.float64] | None = None,
) -> float:
    """Score inconsistency between what an agent claims and what's verifiable.

    Uses both ground truth comparison and consensus from other agents.
    """
    if claimed_output.size == 0 or ground_truth.size == 0:
        return 0.0

    min_dim = min(len(claimed_output), len(ground_truth))
    diff = claimed_output[:min_dim] - ground_truth[:min_dim]
    base_dishonesty = float(np.linalg.norm(diff))

    if other_agents_consensus is not None and other_agents_consensus.size > 0:
        min_dim2 = min(len(claimed_output), len(other_agents_consensus))
        consensus_diff = claimed_output[:min_dim2] - other_agents_consensus[:min_dim2]
        consensus_dishonesty = float(np.linalg.norm(consensus_diff))
        return (base_dishonesty + consensus_dishonesty) / 2.0

    return base_dishonesty


def create_output_stream(agent: Agent, signal: NDArray[np.float64], dim: int) -> Stream:
    """Create an output stream from an agent's compressed signal (for whistleblower consumption)."""
    return Stream(
        stream_type=StreamType.OUTPUT,
        dimensionality=dim,
        source_agent_id=agent.id,
        current_data=signal,
        label=f"output_{agent.id[:8]}",
    )
