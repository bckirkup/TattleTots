"""Sensing and fusion: prepare agent input from multiple streams."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent
from tattletots.models.genome import SensingStrategy
from tattletots.models.stream import Stream


def _stable_sample_indices(
    total_dim: int,
    working_dim: int,
    seed: int,
) -> NDArray[np.int64]:
    """Deterministic subspace sample indices for an agent lineage."""
    rng = np.random.default_rng(seed)
    n = min(working_dim, total_dim)
    return np.sort(rng.choice(total_dim, size=n, replace=False))


def _align_stream(
    data: NDArray[np.float64],
    target_dim: int,
) -> NDArray[np.float64]:
    """Pad or truncate a stream vector to target_dim."""
    if data.size >= target_dim:
        return data[:target_dim].astype(np.float64, copy=False)
    out = np.zeros(target_dim, dtype=np.float64)
    out[: data.size] = data
    return out


def prepare_agent_input(
    agent: Agent,
    streams: dict[str, Stream],
    config: SimulationConfig,
    *,
    spatial_dim_map: dict[str, slice] | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Fuse selected input streams into a working-dimension vector.

    Returns (projected_vector, stream_labels).
    """
    input_data: list[NDArray[np.float64]] = []
    input_labels: list[str] = []

    for stream_id in agent.state.input_stream_ids:
        stream = streams.get(stream_id)
        if stream is not None and stream.current_data.size > 0:
            input_data.append(stream.current_data.astype(np.float64, copy=False))
            input_labels.append(stream.label or stream_id)

    if not input_data:
        return np.array([], dtype=np.float64), input_labels

    genome = agent.genome
    working_dim = min(genome.working_dim, config.max_working_dim, config.max_stream_dim)
    strategy = genome.sensing_strategy

    if strategy == SensingStrategy.CONCAT:
        combined = np.concatenate(input_data)
        if combined.size > working_dim:
            combined = combined[:working_dim]
        elif combined.size < working_dim:
            padded = np.zeros(working_dim, dtype=np.float64)
            padded[: combined.size] = combined
            combined = padded
        return combined, input_labels

    if strategy == SensingStrategy.WEIGHTED_FUSE:
        weights = agent.state.fusion_weights_override
        if weights.size == 0:
            weights = genome.fusion_weights
        if weights.size == 0:
            weights = genome.input_preference
        if weights.size < len(input_data):
            weights = np.ones(len(input_data), dtype=np.float64)
        weights = weights[: len(input_data)]
        weights = weights / max(float(weights.sum()), 1e-10)

        aligned = [_align_stream(d, working_dim) for d in input_data]
        fused = np.zeros(working_dim, dtype=np.float64)
        for w, arr in zip(weights, aligned, strict=True):
            fused += w * arr
        return fused, input_labels

    if strategy == SensingStrategy.SUBSPACE_SAMPLE:
        combined = np.concatenate(input_data)
        seed = hash(agent.id) % (2**31) + genome.dim_offset
        indices = _stable_sample_indices(combined.size, working_dim, seed)
        sampled = combined[indices]
        if sampled.size < working_dim:
            padded = np.zeros(working_dim, dtype=np.float64)
            padded[: sampled.size] = sampled
            return padded, input_labels
        return sampled, input_labels

    if strategy == SensingStrategy.BLOCK_SPECIALIZE:
        combined = np.concatenate(input_data)
        n_blocks = config.n_spatial_blocks
        block_size = max(1, int(np.ceil(combined.size / n_blocks)))
        block_idx = genome.block_index % n_blocks
        start = block_idx * block_size
        end = min(start + block_size, combined.size)
        block = combined[start:end]
        if block.size >= working_dim:
            return block[:working_dim], input_labels
        padded = np.zeros(working_dim, dtype=np.float64)
        padded[: block.size] = block
        return padded, input_labels

    # Fallback
    combined = np.concatenate(input_data)[:working_dim]
    return combined, input_labels


def gather_raw_stream_data(
    agent: Agent,
    streams: dict[str, Stream],
) -> tuple[list[NDArray[np.float64]], list[str]]:
    """Collect raw stream arrays and labels for an agent."""
    input_data: list[NDArray[np.float64]] = []
    input_labels: list[str] = []
    for stream_id in agent.state.input_stream_ids:
        stream = streams.get(stream_id)
        if stream is not None and stream.current_data.size > 0:
            input_data.append(stream.current_data.astype(np.float64, copy=False))
            input_labels.append(stream.label or stream_id)
    return input_data, input_labels
