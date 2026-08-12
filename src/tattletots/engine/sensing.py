"""Sensing and fusion: prepare agent input from multiple streams."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.config import SimulationConfig
from tattletots.engine.identity import stable_id_digest
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, SensingStrategy
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


def _pad_or_truncate(combined: NDArray[np.float64], working_dim: int) -> NDArray[np.float64]:
    if combined.size > working_dim:
        return combined[:working_dim]
    if combined.size < working_dim:
        padded = np.zeros(working_dim, dtype=np.float64)
        padded[: combined.size] = combined
        return padded
    return combined


def _fuse_concat(input_data: list[NDArray[np.float64]], working_dim: int) -> NDArray[np.float64]:
    combined = np.concatenate(input_data)
    return _pad_or_truncate(combined, working_dim)


def _fuse_weighted(
    agent: Agent,
    genome: Genome,
    input_data: list[NDArray[np.float64]],
    working_dim: int,
) -> NDArray[np.float64]:
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
    return fused


def _fuse_subspace_sample(
    agent: Agent,
    genome: Genome,
    input_data: list[NDArray[np.float64]],
    working_dim: int,
) -> NDArray[np.float64]:
    combined = np.concatenate(input_data)
    seed = stable_id_digest(agent.id) % (2**31) + genome.dim_offset
    indices = _stable_sample_indices(combined.size, working_dim, seed)
    sampled = combined[indices]
    return _pad_or_truncate(sampled, working_dim)


def _fuse_block_specialize(
    genome: Genome,
    input_data: list[NDArray[np.float64]],
    working_dim: int,
    n_blocks: int,
) -> NDArray[np.float64]:
    combined = np.concatenate(input_data)
    block_size = max(1, int(np.ceil(combined.size / n_blocks)))
    block_idx = genome.block_index % n_blocks
    start = block_idx * block_size
    end = min(start + block_size, combined.size)
    block = combined[start:end]
    return _pad_or_truncate(block, working_dim)


def prepare_agent_input(
    agent: Agent,
    streams: dict[str, Stream],
    config: SimulationConfig,
    *,
    _spatial_dim_map: dict[str, slice] | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Fuse selected input streams into a working-dimension vector.

    Returns (projected_vector, stream_labels).
    """
    projected, labels, _, _ = prepare_agent_input_with_attribution(
        agent,
        streams,
        config,
        _spatial_dim_map=_spatial_dim_map,
    )
    return projected, labels


def prepare_agent_input_with_attribution(
    agent: Agent,
    streams: dict[str, Stream],
    config: SimulationConfig,
    *,
    _spatial_dim_map: dict[str, slice] | None = None,
) -> tuple[NDArray[np.float64], list[str], float, float]:
    """Fuse inputs and return grounded/ungrounded input-dimension mass.

    Grounded mass comes from raw domain streams. All non-raw streams, including
    curated residuals, contribute to the ungrounded mass.
    """
    input_data: list[NDArray[np.float64]] = []
    input_labels: list[str] = []
    input_streams: list[Stream] = []

    for stream_id in agent.state.input_stream_ids:
        stream = streams.get(stream_id)
        if stream is not None and stream.current_data.size > 0:
            input_data.append(stream.current_data.astype(np.float64, copy=False))
            input_labels.append(stream.label or stream_id)
            input_streams.append(stream)

    if not input_data:
        return np.array([], dtype=np.float64), input_labels, 0.0, 0.0

    genome = agent.genome
    working_dim = min(genome.working_dim, config.max_working_dim, config.max_stream_dim)
    strategy = genome.sensing_strategy

    if strategy == SensingStrategy.CONCAT:
        projected = _fuse_concat(input_data, working_dim)
        grounded, ungrounded = _mass_for_concat(input_streams, working_dim)
        return projected, input_labels, grounded, ungrounded

    if strategy == SensingStrategy.WEIGHTED_FUSE:
        projected = _fuse_weighted(agent, genome, input_data, working_dim)
        grounded, ungrounded = _mass_for_weighted(agent, genome, input_streams, working_dim)
        return projected, input_labels, grounded, ungrounded

    if strategy == SensingStrategy.SUBSPACE_SAMPLE:
        projected = _fuse_subspace_sample(agent, genome, input_data, working_dim)
        grounded, ungrounded = _mass_for_subspace(agent, genome, input_streams, working_dim)
        return projected, input_labels, grounded, ungrounded

    if strategy == SensingStrategy.BLOCK_SPECIALIZE:
        projected = _fuse_block_specialize(
            genome,
            input_data,
            working_dim,
            config.n_spatial_blocks,
        )
        grounded, ungrounded = _mass_for_block(
            genome,
            input_streams,
            working_dim,
            config.n_spatial_blocks,
        )
        return projected, input_labels, grounded, ungrounded

    combined = np.concatenate(input_data)[:working_dim]
    grounded, ungrounded = _mass_for_concat(input_streams, working_dim)
    return combined, input_labels, grounded, ungrounded


def _mass_for_concat(streams: list[Stream], working_dim: int) -> tuple[float, float]:
    grounded = 0.0
    ungrounded = 0.0
    remaining = working_dim
    for stream in streams:
        contribution = min(stream.current_data.size, remaining)
        if stream.stream_type == "raw":
            grounded += contribution
        else:
            ungrounded += contribution
        remaining -= contribution
        if remaining <= 0:
            break
    return grounded, ungrounded


def _weighted_masses(
    agent: Agent,
    genome: Genome,
    streams: list[Stream],
) -> list[float]:
    weights = agent.state.fusion_weights_override
    if weights.size == 0:
        weights = genome.fusion_weights
    if weights.size == 0:
        weights = genome.input_preference
    if weights.size < len(streams):
        weights = np.ones(len(streams), dtype=np.float64)
    weights = weights[: len(streams)]
    normalized = weights / max(float(weights.sum()), 1e-10)
    return [float(weight) for weight in normalized]


def _mass_for_weighted(
    agent: Agent,
    genome: Genome,
    streams: list[Stream],
    working_dim: int,
) -> tuple[float, float]:
    grounded = 0.0
    ungrounded = 0.0
    for stream, weight in zip(streams, _weighted_masses(agent, genome, streams), strict=True):
        contribution = weight * working_dim
        if stream.stream_type == "raw":
            grounded += contribution
        else:
            ungrounded += contribution
    return grounded, ungrounded


def _source_for_dimensions(streams: list[Stream]) -> list[bool]:
    source_is_grounded: list[bool] = []
    for stream in streams:
        source_is_grounded.extend([stream.stream_type == "raw"] * stream.current_data.size)
    return source_is_grounded


def _mass_for_subspace(
    agent: Agent,
    genome: Genome,
    streams: list[Stream],
    working_dim: int,
) -> tuple[float, float]:
    source_is_grounded = _source_for_dimensions(streams)
    total_dim = len(source_is_grounded)
    n = min(working_dim, total_dim)
    seed = stable_id_digest(agent.id) % (2**31) + genome.dim_offset
    indices = _stable_sample_indices(total_dim, working_dim, seed)
    grounded = sum(source_is_grounded[int(index)] for index in indices[:n])
    return float(grounded), float(n - grounded)


def _mass_for_block(
    genome: Genome,
    streams: list[Stream],
    working_dim: int,
    n_blocks: int,
) -> tuple[float, float]:
    source_is_grounded = _source_for_dimensions(streams)
    block_size = max(1, int(np.ceil(len(source_is_grounded) / n_blocks)))
    block_idx = genome.block_index % n_blocks
    start = block_idx * block_size
    end = min(start + block_size, len(source_is_grounded))
    selected = source_is_grounded[start:end]
    selected = selected[:working_dim]
    grounded = sum(selected)
    return float(grounded), float(len(selected) - grounded)


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
