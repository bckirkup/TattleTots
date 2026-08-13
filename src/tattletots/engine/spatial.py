"""Spatial region specialization for compression and reporting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, SpatialInferenceStrategy, SpatialStrategy
from tattletots.models.identity import stable_id_digest
from tattletots.models.location import EventLocation, LocationFrame
from tattletots.models.observation import ObservationPacket, ObservationStatus

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


def apply_spatial_observation(
    agent: Agent,
    observation: ObservationPacket,
    *,
    n_blocks: int = 10,
    dim_to_location: DimToLocationFn | None = None,
) -> ObservationPacket:
    """Apply spatial weighting and mark masked features as unavailable."""
    masked = apply_spatial_mask(
        agent,
        observation.data,
        n_blocks=n_blocks,
        dim_to_location=dim_to_location,
    )
    if observation.status is None or observation.status.size == 0:
        return ObservationPacket(
            masked,
            observation.metadata,
            observation.status,
            observation.observed_fraction,
        )
    status = observation.status.copy()
    mask = agent.state.last_spatial_mask
    if mask.size == status.size:
        status[np.isclose(mask, 0.0, rtol=0.0, atol=0.0)] = ObservationStatus.MASKED.value
    return ObservationPacket(
        masked,
        observation.metadata,
        status,
        observation.observed_fraction,
    )


def _feature_evidence(
    genome: Genome,
    observation: ObservationPacket,
    index: int,
) -> float:
    """Convert one feature into generic spatial evidence."""
    reliability = 1.0
    metadata = observation.metadata
    if metadata is not None and metadata.modality is not None:
        modality = metadata.modality[index]
        if modality is not None and genome.modality_reliability.size > 0:
            bucket = stable_id_digest(modality) % genome.modality_reliability.size
            reliability = float(genome.modality_reliability[bucket])

    availability = 1.0
    if observation.observed_fraction is not None:
        availability = float(observation.observed_fraction[index])
    if observation.status is not None:
        status = observation.status[index]
        if status == ObservationStatus.MASKED.value:
            return 0.0
        if status == ObservationStatus.MISSING.value:
            availability = 0.0
    return max(
        0.0,
        reliability * abs(float(observation.data[index]))
        + genome.absence_weight * (1.0 - availability),
    )


def _project_location_to_observed_hull(
    location: EventLocation, coordinates: NDArray[np.float64]
) -> EventLocation:
    """Project a heritable prior into the coordinate hull supplied by evidence."""
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    projected = np.clip(np.asarray(location, dtype=np.float64), lower, upper)
    return (int(round(projected[0])), int(round(projected[1])))


def project_location_to_frame(location: EventLocation, frame: LocationFrame) -> EventLocation:
    """Project a report into a domain's declared public coordinate frame."""
    lower, upper = frame
    projected = np.clip(
        np.asarray(location, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    return (int(round(projected[0])), int(round(projected[1])))


def infer_geometry_location(
    agent: Agent,
    observation: ObservationPacket,
) -> EventLocation | None:
    """Decode a coordinate-bearing observation using heritable generic traits."""
    metadata = observation.metadata
    if metadata is None or metadata.coordinates is None:
        return None
    if metadata.feature_count != observation.data.size:
        return None

    points = [
        (index, coordinate)
        for index, coordinate in enumerate(metadata.coordinates)
        if coordinate is not None and len(coordinate) >= 2
    ]
    if not points:
        return None

    genome = agent.genome
    evidence = np.array(
        [_feature_evidence(genome, observation, index) for index, _ in points],
        dtype=np.float64,
    )
    coordinates = np.asarray(
        [[point[0], point[1]] for _, point in points],
        dtype=np.float64,
    )
    if not np.any(evidence > 0.0):
        return _project_location_to_observed_hull(genome.spatial_region, coordinates)

    if genome.spatial_inference_strategy == SpatialInferenceStrategy.FIXED_PRIOR:
        return _project_location_to_observed_hull(genome.spatial_region, coordinates)
    if genome.spatial_inference_strategy == SpatialInferenceStrategy.PEAK:
        selected = coordinates[int(np.argmax(evidence))]
        return (int(round(selected[0])), int(round(selected[1])))

    if genome.spatial_inference_strategy == SpatialInferenceStrategy.WEIGHTED_CENTROID:
        centroid = np.average(coordinates, axis=0, weights=evidence)
        return (int(round(centroid[0])), int(round(centroid[1])))

    low = np.floor(coordinates.min(axis=0)).astype(int)
    high = np.ceil(coordinates.max(axis=0)).astype(int)
    candidates = np.array(
        [(row, col) for row in range(low[0], high[0] + 1) for col in range(low[1], high[1] + 1)],
        dtype=np.float64,
    )
    distances = candidates[:, None, :] - coordinates[None, :, :]
    squared_distance = np.sum(distances * distances, axis=2)
    bandwidth = max(genome.spatial_kernel_bandwidth, 0.1)
    kernel = np.exp(-squared_distance / (2.0 * bandwidth**2))
    power = max(genome.spatial_distance_power, 0.0)
    if power > 0.0:
        kernel /= (1.0 + np.sqrt(squared_distance)) ** power
    scores = kernel @ evidence
    selected = candidates[int(np.argmax(scores))]
    return (int(selected[0]), int(selected[1]))


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
