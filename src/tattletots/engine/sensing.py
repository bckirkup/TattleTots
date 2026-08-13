"""Sensing and fusion: prepare agent input from multiple streams."""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, SensingStrategy
from tattletots.models.identity import stable_id_digest
from tattletots.models.observation import ObservationPacket, ObservationStatus, StreamMetadata
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
    input_data: list[NDArray[np.float64]],
    working_dim: int,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    combined = np.concatenate(input_data)
    sampled = combined[indices]
    return _pad_or_truncate(sampled, working_dim)


def _fuse_block_specialize(
    input_data: list[NDArray[np.float64]],
    working_dim: int,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    combined = np.concatenate(input_data)
    block = combined[indices]
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
    projected, input_labels, grounded, ungrounded, _ = _prepare_agent_input_with_selection(
        agent,
        streams,
        config,
    )
    return projected, input_labels, grounded, ungrounded


def _prepare_agent_input_with_selection(
    agent: Agent,
    streams: dict[str, Stream],
    config: SimulationConfig,
) -> tuple[
    NDArray[np.float64],
    list[str],
    float,
    float,
    NDArray[np.int64] | None,
]:
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
        return np.array([], dtype=np.float64), input_labels, 0.0, 0.0, None

    genome = agent.genome
    working_dim = min(genome.working_dim, config.max_working_dim, config.max_stream_dim)
    strategy = genome.sensing_strategy
    selection: NDArray[np.int64] | None = None

    if strategy == SensingStrategy.CONCAT:
        projected = _fuse_concat(input_data, working_dim)
        grounded, ungrounded = _mass_for_concat(input_streams, working_dim)
        selection = np.arange(
            min(sum(data.size for data in input_data), working_dim),
            dtype=np.int64,
        )
        return projected, input_labels, grounded, ungrounded, selection

    if strategy == SensingStrategy.WEIGHTED_FUSE:
        projected = _fuse_weighted(agent, genome, input_data, working_dim)
        grounded, ungrounded = _mass_for_weighted(agent, genome, input_streams, working_dim)
        return projected, input_labels, grounded, ungrounded, None

    if strategy == SensingStrategy.SUBSPACE_SAMPLE:
        total_dim = sum(data.size for data in input_data)
        seed = stable_id_digest(agent.id) % (2**31) + genome.dim_offset
        selection = _stable_sample_indices(total_dim, working_dim, seed)
        projected = _fuse_subspace_sample(input_data, working_dim, selection)
        grounded, ungrounded = _mass_for_subspace(
            input_streams,
            selection,
        )
        return projected, input_labels, grounded, ungrounded, selection

    if strategy == SensingStrategy.BLOCK_SPECIALIZE:
        total_dim = sum(data.size for data in input_data)
        block_size = max(1, int(np.ceil(total_dim / config.n_spatial_blocks)))
        block_idx = genome.block_index % config.n_spatial_blocks
        start = block_idx * block_size
        end = min(start + block_size, total_dim)
        selection = np.arange(
            start,
            min(end, start + working_dim),
            dtype=np.int64,
        )
        projected = _fuse_block_specialize(input_data, working_dim, selection)
        grounded, ungrounded = _mass_for_block(input_streams, selection)
        return projected, input_labels, grounded, ungrounded, selection

    combined = np.concatenate(input_data)[:working_dim]
    grounded, ungrounded = _mass_for_concat(input_streams, working_dim)
    selection = np.arange(min(combined.size, working_dim), dtype=np.int64)
    return combined, input_labels, grounded, ungrounded, selection


def prepare_agent_observation(
    agent: Agent,
    streams: dict[str, Stream],
    config: SimulationConfig,
) -> tuple[ObservationPacket, list[str], float, float]:
    """Prepare numeric input and transport metadata for one agent.

    Existing callers can continue using ``prepare_agent_input``.  This
    metadata-bearing path is deliberately separate so transport does not
    alter legacy numeric behavior.
    """
    projected, labels, grounded, ungrounded, selection = _prepare_agent_input_with_selection(
        agent,
        streams,
        config,
    )
    input_streams = [
        streams[stream_id]
        for stream_id in agent.state.input_stream_ids
        if stream_id in streams and streams[stream_id].current_data.size > 0
    ]
    if not input_streams or projected.size == 0:
        return ObservationPacket(data=projected), labels, grounded, ungrounded

    metadata = _combined_stream_metadata(input_streams)
    if metadata is None:
        return ObservationPacket(data=projected), labels, grounded, ungrounded

    statuses = _combined_stream_status(input_streams)
    working_dim = min(agent.genome.working_dim, config.max_working_dim, config.max_stream_dim)
    if selection is None:
        # Weighted addition destroys one-to-one feature provenance.
        return ObservationPacket(data=projected), labels, grounded, ungrounded
    metadata = metadata.select(selection).truncate_or_pad(working_dim)
    statuses = _pad_status(statuses[selection], working_dim)
    return ObservationPacket(projected, metadata, statuses), labels, grounded, ungrounded


def _combined_stream_metadata(streams: list[Stream]) -> StreamMetadata | None:
    if not any(stream.metadata is not None for stream in streams):
        return None
    coordinates = _combined_metadata_field(streams, "coordinates")
    modalities = _combined_metadata_field(streams, "modality")
    identities = _combined_metadata_field(streams, "identity")
    footprints = _combined_metadata_field(streams, "footprints")
    resolutions = _combined_metadata_field(streams, "resolution")
    return StreamMetadata(
        coordinates=coordinates,
        modality=modalities,
        identity=identities,
        footprints=footprints,
        resolution=resolutions,
    )


@overload
def _combined_metadata_field(
    streams: list[Stream], field: Literal["coordinates", "footprints"]
) -> list[tuple[float, ...] | None]: ...


@overload
def _combined_metadata_field(
    streams: list[Stream], field: Literal["modality", "identity"]
) -> list[str | None]: ...


@overload
def _combined_metadata_field(
    streams: list[Stream], field: Literal["resolution"]
) -> list[float | None]: ...


def _combined_metadata_field(streams: list[Stream], field: str) -> list[Any]:
    values: list[Any] = []
    for stream in streams:
        metadata = stream.metadata
        field_values = getattr(metadata, field, None) if metadata is not None else None
        values.extend(field_values if field_values is not None else [None] * stream.dimensionality)
    return values


def _combined_stream_status(streams: list[Stream]) -> NDArray[np.str_]:
    statuses: list[str] = []
    for stream in streams:
        if stream.current_status.size == stream.dimensionality:
            statuses.extend(str(item) for item in stream.current_status)
        else:
            statuses.extend([ObservationStatus.OBSERVED.value] * stream.dimensionality)
    return np.asarray(statuses, dtype="<U8")


def _pad_status(status: NDArray[np.str_], dimensionality: int) -> NDArray[np.str_]:
    if status.size >= dimensionality:
        return status[:dimensionality]
    return np.concatenate(
        [
            status,
            np.full(dimensionality - status.size, ObservationStatus.MISSING.value, dtype="<U8"),
        ]
    )


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
    streams: list[Stream],
    indices: NDArray[np.int64],
) -> tuple[float, float]:
    source_is_grounded = _source_for_dimensions(streams)
    grounded = sum(source_is_grounded[int(index)] for index in indices)
    return float(grounded), float(len(indices) - grounded)


def _mass_for_block(
    streams: list[Stream],
    indices: NDArray[np.int64],
) -> tuple[float, float]:
    source_is_grounded = _source_for_dimensions(streams)
    selected = [source_is_grounded[int(index)] for index in indices]
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
