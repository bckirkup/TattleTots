"""Spatial region specialization for compression and reporting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import SpatialStrategy
from tattletots.models.location import EventLocation

DimToLocationFn = Callable[[int], EventLocation]


def apply_spatial_mask(
    agent: Agent,
    data: NDArray[np.float64],
    *,
    n_blocks: int = 10,
    dim_to_location: DimToLocationFn | None = None,
) -> NDArray[np.float64]:
    """Apply spatial specialization mask to input vector."""
    if data.size == 0:
        agent.state.last_spatial_mask = np.array([], dtype=np.float64)
        return data

    genome = agent.genome
    strategy = genome.spatial_strategy
    mask = np.ones(data.size, dtype=np.float64)

    if strategy == SpatialStrategy.GLOBAL:
        agent.state.last_spatial_mask = mask
        return data

    if strategy == SpatialStrategy.PEAK:
        peak_idx = int(np.argmax(np.abs(data)))
        mask[:] = 0.0
        lo = max(0, peak_idx - 1)
        hi = min(data.size, peak_idx + 2)
        mask[lo:hi] = 1.0

    elif strategy == SpatialStrategy.WEIGHTED_ROI:
        aff = genome.region_affinity
        if aff.size >= data.size:
            mask = aff[: data.size].copy()
        elif aff.size > 0:
            # Tile affinity across dimensions
            reps = int(np.ceil(data.size / aff.size))
            tiled = np.tile(aff, reps)[: data.size]
            mask = tiled / max(float(tiled.sum()), 1e-10)
        else:
            block_size = max(1, data.size // n_blocks)
            block_idx = genome.block_index % n_blocks
            start = block_idx * block_size
            end = min(start + block_size, data.size)
            mask[:] = 0.0
            mask[start:end] = 1.0

    elif strategy == SpatialStrategy.FIXED_REGION:
        if dim_to_location is not None:
            target = genome.spatial_region
            radius = genome.spatial_radius
            for i in range(data.size):
                loc = dim_to_location(i)
                dist = abs(loc[0] - target[0]) + abs(loc[1] - target[1])
                mask[i] = 1.0 if dist <= radius else 0.0
        else:
            block_size = max(1, data.size // n_blocks)
            center = genome.spatial_region[0] % n_blocks
            start = center * block_size
            end = min(start + block_size * (genome.spatial_radius + 1), data.size)
            mask[:] = 0.0
            mask[start:end] = 1.0

    agent.state.last_spatial_mask = mask
    return data * mask


def infer_spatial_location(
    agent: Agent,
    data: NDArray[np.float64],
    *,
    n_blocks: int = 10,
    dim_to_location: DimToLocationFn | None = None,
) -> EventLocation:
    """Infer report location from spatially weighted input."""
    if data.size == 0:
        return (0, 0)

    genome = agent.genome
    strategy = genome.spatial_strategy

    if strategy == SpatialStrategy.GLOBAL:
        peak_idx = int(np.argmax(np.abs(data)))
        if dim_to_location is not None:
            return dim_to_location(peak_idx)
        return (peak_idx % n_blocks, 0)

    if strategy == SpatialStrategy.FIXED_REGION:
        return genome.spatial_region

    # PEAK and WEIGHTED_ROI: centroid of top activations
    mask = agent.state.last_spatial_mask
    if mask.size != data.size:
        mask = np.ones(data.size, dtype=np.float64)
    weighted = np.abs(data) * mask
    if weighted.sum() <= 0:
        return genome.spatial_region

    peak_idx = int(np.argmax(weighted))
    if dim_to_location is not None:
        return dim_to_location(peak_idx)

    block_size = max(1, data.size // n_blocks)
    block = peak_idx // block_size
    return (block, 0)
