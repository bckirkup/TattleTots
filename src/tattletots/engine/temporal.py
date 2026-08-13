"""Temporal memory fusion before compression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import TemporalFusionMode
from tattletots.models.observation import ObservationPacket


def _fuse_ema(buffer: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    alpha = 2.0 / (len(buffer) + 1)
    ema = buffer[0].copy()
    for sample in buffer[1:]:
        ema = (1 - alpha) * ema + alpha * sample
    return ema


def _fuse_window_stack(buffer: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    window = np.stack(buffer, axis=0)
    result: NDArray[np.float64] = window.mean(axis=0)
    return result


def _fuse_ar_lag(
    buffer: list[NDArray[np.float64]],
    current: NDArray[np.float64],
) -> NDArray[np.float64]:
    prev = buffer[-2]
    if prev.shape != current.shape:
        return current
    denom = float(np.dot(prev, prev))
    if denom < 1e-10:
        return current
    coeff = float(np.dot(current, prev)) / denom
    predicted = coeff * prev
    return 0.5 * current + 0.5 * (current - predicted)


def apply_temporal_fusion(
    agent: Agent,
    current: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fuse current sensing output with temporal history buffer.

    Updates agent.state.temporal_buffer in place.
    """
    if current.size == 0:
        return current

    genome = agent.genome
    depth = genome.temporal_memory_depth
    mode = genome.temporal_fusion_mode

    if depth <= 0 or mode == TemporalFusionMode.NONE:
        return current

    buffer = agent.state.temporal_buffer
    buffer.append(current.copy())
    if len(buffer) > depth:
        agent.state.temporal_buffer = buffer[-depth:]
        buffer = agent.state.temporal_buffer

    if mode == TemporalFusionMode.EMA:
        return _fuse_ema(buffer)

    if mode == TemporalFusionMode.WINDOW_STACK:
        if len(buffer) < 2:
            return current
        return _fuse_window_stack(buffer)

    if mode == TemporalFusionMode.AR_LAG:
        if len(buffer) < 2:
            return current
        return _fuse_ar_lag(buffer, current)

    return current


def apply_temporal_observation(agent: Agent, current: ObservationPacket) -> ObservationPacket:
    """Fuse an observation while retaining geometry only when it is stable."""
    if current.data.size == 0:
        return current
    if current.metadata is None:
        return ObservationPacket(apply_temporal_fusion(agent, current.data))

    # The runtime history stores numeric arrays only.  Geometry is valid after
    # fusion only when every retained sample has the same feature schema.
    fused = apply_temporal_fusion(agent, current.data)
    if len(agent.state.temporal_buffer) > 1:
        # Consequently temporal accumulation of spatial evidence is not yet
        # expressible; a later slice must carry feature schema in the history.
        return ObservationPacket(fused)
    return ObservationPacket(fused, current.metadata, current.status)
