"""Temporal memory fusion before compression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import TemporalFusionMode
from tattletots.models.observation import ObservationPacket, ObservationStatus, StreamMetadata


def _fuse_ema(buffer: list[ObservationPacket]) -> NDArray[np.float64]:
    alpha = 2.0 / (len(buffer) + 1)
    ema = buffer[0].data.copy()
    for sample in buffer[1:]:
        ema = (1 - alpha) * ema + alpha * sample.data
    return ema


def _fuse_window_stack(buffer: list[ObservationPacket]) -> NDArray[np.float64]:
    window = np.stack([sample.data for sample in buffer], axis=0)
    result: NDArray[np.float64] = window.mean(axis=0)
    return result


def _fuse_ar_lag(
    buffer: list[ObservationPacket],
    current: NDArray[np.float64],
) -> NDArray[np.float64]:
    prev = buffer[-2].data
    if prev.shape != current.shape:
        return current
    denom = float(np.dot(prev, prev))
    if denom < 1e-10:
        return current
    coeff = float(np.dot(current, prev)) / denom
    predicted = coeff * prev
    return 0.5 * current + 0.5 * (current - predicted)


def _append_sample(agent: Agent, current: ObservationPacket) -> list[ObservationPacket]:
    sample = ObservationPacket(
        data=current.data.copy(),
        metadata=current.metadata,
        status=None if current.status is None else current.status.copy(),
    )
    buffer = agent.state.temporal_buffer
    buffer.append(sample)
    depth = agent.genome.temporal_memory_depth
    if len(buffer) > depth:
        agent.state.temporal_buffer = buffer[-depth:]
        return agent.state.temporal_buffer
    return buffer


def _has_shape_change(buffer: list[ObservationPacket], current: ObservationPacket) -> bool:
    return any(sample.data.shape != current.data.shape for sample in buffer)


def _fuse_packet(agent: Agent, current: ObservationPacket) -> ObservationPacket:
    if current.data.size == 0:
        return current

    genome = agent.genome
    depth = genome.temporal_memory_depth
    mode = genome.temporal_fusion_mode

    if depth <= 0 or mode == TemporalFusionMode.NONE:
        return current

    shape_changed = _has_shape_change(agent.state.temporal_buffer, current)
    if shape_changed:
        # Dimensionality changes invalidate temporal fusion, matching the
        # compression models' reset-on-dimension-change behavior.
        agent.state.temporal_buffer = []
    buffer = _append_sample(agent, current)
    if shape_changed:
        return ObservationPacket(data=current.data)

    if mode == TemporalFusionMode.EMA:
        fused = _fuse_ema(buffer)
    elif mode == TemporalFusionMode.WINDOW_STACK:
        fused = current.data if len(buffer) < 2 else _fuse_window_stack(buffer)
    elif mode == TemporalFusionMode.AR_LAG:
        fused = current.data if len(buffer) < 2 else _fuse_ar_lag(buffer, current.data)
    else:
        fused = current.data
    shared_metadata = _shared_metadata(buffer, fused.size)
    return ObservationPacket(
        data=fused,
        metadata=shared_metadata,
        status=_fused_status(buffer, fused.size),
        observed_fraction=(
            _observed_fraction(buffer, fused.size) if shared_metadata is not None else None
        ),
    )


def apply_temporal_fusion(
    agent: Agent,
    current: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fuse current sensing output with temporal history buffer.

    Updates agent.state.temporal_buffer in place while preserving the legacy
    numeric-only API.
    """
    return _fuse_packet(agent, ObservationPacket(data=current)).data


def _shared_metadata(
    buffer: list[ObservationPacket],
    dimensionality: int,
) -> StreamMetadata | None:
    metadata = buffer[0].metadata
    if metadata is None or metadata.feature_count != dimensionality:
        return None
    if any(sample.metadata != metadata for sample in buffer[1:]):
        return None
    return metadata


def _sample_status(sample: ObservationPacket) -> NDArray[np.str_] | None:
    if sample.status is None or sample.status.size == 0:
        return None
    return sample.status


def _fused_status(
    buffer: list[ObservationPacket],
    dimensionality: int,
) -> NDArray[np.str_] | None:
    statuses = [_sample_status(sample) for sample in buffer]
    if any(status is None or status.size != dimensionality for status in statuses):
        return None
    result = np.full(dimensionality, ObservationStatus.OBSERVED.value, dtype="<U8")
    missing = np.zeros(dimensionality, dtype=bool)
    masked = np.zeros(dimensionality, dtype=bool)
    for status in statuses:
        assert status is not None
        missing |= status == ObservationStatus.MISSING.value
        masked |= status == ObservationStatus.MASKED.value
    result[missing] = ObservationStatus.MISSING.value
    result[masked] = ObservationStatus.MASKED.value
    return result


def _observed_fraction(
    buffer: list[ObservationPacket],
    dimensionality: int,
) -> NDArray[np.float64] | None:
    statuses = [_sample_status(sample) for sample in buffer]
    if any(status is None or status.size != dimensionality for status in statuses):
        return None
    observed = np.zeros(dimensionality, dtype=np.float64)
    for status in statuses:
        assert status is not None
        observed += status == ObservationStatus.OBSERVED.value
    return observed / len(statuses)


def apply_temporal_observation(agent: Agent, current: ObservationPacket) -> ObservationPacket:
    """Fuse an observation while retaining only genuinely stable geometry."""
    fused = _fuse_packet(agent, current)
    if current.metadata is None and current.status is None:
        return ObservationPacket(fused.data)
    return fused
