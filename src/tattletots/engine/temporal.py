"""Temporal memory fusion before compression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import TemporalFusionMode


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
        alpha = 2.0 / (len(buffer) + 1)
        ema = buffer[0].copy()
        for sample in buffer[1:]:
            ema = (1 - alpha) * ema + alpha * sample
        return ema

    if mode == TemporalFusionMode.WINDOW_STACK:
        if len(buffer) < 2:
            return current
        window = np.stack(buffer, axis=0)
        # Lightweight projection: mean across time preserves dim, reduces noise
        result: NDArray[np.float64] = window.mean(axis=0)
        return result

    if mode == TemporalFusionMode.AR_LAG:
        if len(buffer) < 2:
            return current
        prev = buffer[-2]
        if prev.shape != current.shape:
            return current
        denom = float(np.dot(prev, prev))
        if denom < 1e-10:
            return current
        coeff = float(np.dot(current, prev)) / denom
        predicted = coeff * prev
        # Return residual-like temporal signal fused with current
        return 0.5 * current + 0.5 * (current - predicted)

    return current
