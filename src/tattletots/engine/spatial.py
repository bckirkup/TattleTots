"""Spatial region specialization for compression and reporting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, SpatialStrategy
from tattletots.models.location import EventLocation

DimToLocationFn = Callable[[int], EventLocation]


def _peak_mask(data: NDArray[np.float64]) -> NDArray[np.float64]:
    mask = np.zeros(data.size, dtype=np.float64)
    peak_idx = int(np.argmax(np.abs(data)))
    lo = max(0, peak_idx - 1)
    hi = min(data.size, peak_idx + 2)
    mask[lo:hi] = 1.0
    return mask


def _weighted_roi_mask(
    genome: Genome, data: NDArray[np.float64], n_blocks: int
) -> NDArray[np.float64]:
    aff = genome.region_affinity
    if aff.size >= data.size:
        return aff[: data.size].copy()
    if aff.size > 0:
        reps = int(np.ceil(data.size / aff.size))
        tiled = np.tile(aff, reps)[: data.size]
        return tiled / max(float(tiled.sum()), 1e-10)
    block_size = max(1, data.size // n_blocks)
    block_idx = genome.block_index % n_blocks
    start = block_idx * block_size
    end = min(start + block_size, data.size)
    mask = np.zeros(data.size, dtype=np.float64)
    mask[start:end] = 1.0
    return mask


def _fixed_region_mask(
    genome: Genome,
    data: NDArray[np.float64],
    *,
    n_blocks: int,
    dim_to_location: DimToLocationFn | None,
) -> NDArray[np.float64]:
    mask = np.ones(data.size, dtype=np.float64)
    if dim_to_location is not None:
        target = genome.spatial_region
        radius = genome.spatial_radius
        for i in range(data.size):
            loc = dim_to_location(i)
            dist = abs(loc[0] - target[0]) + abs(loc[1] - target[1])
            mask[i] = 1.0 if dist <= radius else 0.0
        return mask
    block_size = max(1, data.size // n_blocks)
    center = genome.spatial_region[0] % n_blocks
    start = center * block_size
    end = min(start + block_size * (genome.spatial_radius + 1), data.size)
    mask[:] = 0.0
    mask[start:end] = 1.0
    return mask


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
        mask = _peak_mask(data)
    elif strategy == SpatialStrategy.WEIGHTED_ROI:
        mask = _weighted_roi_mask(genome, data, n_blocks)
    elif strategy == SpatialStrategy.FIXED_REGION:
        mask = _fixed_region_mask(genome, data, n_blocks=n_blocks, dim_to_location=dim_to_location)

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
