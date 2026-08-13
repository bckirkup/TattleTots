"""Domain-neutral metadata attached to stream observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class ObservationStatus(StrEnum):
    """Availability state for one published feature."""

    OBSERVED = "observed"
    MISSING = "missing"
    MASKED = "masked"


class StreamMetadata(BaseModel):
    """Optional per-feature interpretation metadata supplied by a domain.

    ``None`` entries deliberately mean that the corresponding feature has no
    declared value for that metadata dimension.  The engine transports these
    declarations but does not interpret domain concepts.
    """

    coordinates: list[tuple[float, ...] | None] | None = Field(
        default=None,
        description="Coordinate of each observed object, when the domain declares one.",
    )
    sensor_coordinates: list[tuple[float, ...] | None] | None = Field(
        default=None,
        description="Static sensor geometry for each feature, when publicly known.",
    )
    modality: list[str | None] | None = Field(
        default=None,
        description="Modality name for each feature.",
    )
    identity: list[str | None] | None = Field(
        default=None,
        description="Identity of the observed object for each feature, when any.",
    )
    footprints: list[tuple[float, ...] | None] | None = Field(
        default=None,
        description="Spatial footprint or support of each feature.",
    )
    resolution: list[float | None] | None = Field(
        default=None,
        description="Spatial resolution associated with each feature.",
    )

    def validate_dimensionality(self, dimensionality: int) -> None:
        """Validate that every declared per-feature field matches a stream."""
        for name in (
            "coordinates",
            "sensor_coordinates",
            "modality",
            "identity",
            "footprints",
            "resolution",
        ):
            values = getattr(self, name)
            if values is not None and len(values) != dimensionality:
                raise ValueError(
                    f"Metadata field {name!r} has {len(values)} entries; expected {dimensionality}"
                )

    def select(self, indices: NDArray[np.int64]) -> StreamMetadata:
        """Select feature metadata using the same indices as numeric sensing."""
        return type(self)(
            coordinates=(
                None if self.coordinates is None else [self.coordinates[int(i)] for i in indices]
            ),
            sensor_coordinates=(
                None
                if self.sensor_coordinates is None
                else [self.sensor_coordinates[int(i)] for i in indices]
            ),
            modality=None if self.modality is None else [self.modality[int(i)] for i in indices],
            identity=None if self.identity is None else [self.identity[int(i)] for i in indices],
            footprints=(
                None if self.footprints is None else [self.footprints[int(i)] for i in indices]
            ),
            resolution=(
                None if self.resolution is None else [self.resolution[int(i)] for i in indices]
            ),
        )

    def truncate_or_pad(self, dimensionality: int) -> StreamMetadata:
        """Apply numeric truncation/padding without inventing geometry."""
        indices = np.arange(min(self.feature_count, dimensionality), dtype=np.int64)
        selected = self.select(indices)
        if selected.feature_count < dimensionality:
            pad = dimensionality - selected.feature_count
            if selected.coordinates is not None:
                selected.coordinates.extend([None] * pad)
            if selected.sensor_coordinates is not None:
                selected.sensor_coordinates.extend([None] * pad)
            if selected.modality is not None:
                selected.modality.extend([None] * pad)
            if selected.identity is not None:
                selected.identity.extend([None] * pad)
            if selected.footprints is not None:
                selected.footprints.extend([None] * pad)
            if selected.resolution is not None:
                selected.resolution.extend([None] * pad)
        return selected

    @property
    def feature_count(self) -> int:
        """Number of features represented by this metadata."""
        for name in (
            "coordinates",
            "sensor_coordinates",
            "modality",
            "identity",
            "footprints",
            "resolution",
        ):
            values = getattr(self, name)
            if values is not None:
                return len(values)
        return 0


@dataclass(frozen=True)
class ObservationPacket:
    """Numeric observation plus transport metadata for one pipeline stage."""

    data: NDArray[np.float64]
    metadata: StreamMetadata | None = None
    status: NDArray[np.str_] | None = None
    observed_fraction: NDArray[np.float64] | None = None

    @classmethod
    def from_stream(
        cls,
        data: NDArray[np.float64],
        metadata: StreamMetadata | None,
        status: NDArray[np.str_] | None,
    ) -> ObservationPacket:
        if status is None or status.size == 0:
            status = np.full(data.size, ObservationStatus.OBSERVED.value, dtype="<U8")
        return cls(data=data, metadata=metadata, status=status)

    def with_data(
        self,
        data: NDArray[np.float64],
        *,
        metadata: StreamMetadata | None = None,
        status: NDArray[np.str_] | None = None,
    ) -> ObservationPacket:
        return ObservationPacket(
            data=data,
            metadata=self.metadata if metadata is None else metadata,
            status=self.status if status is None else status,
            observed_fraction=self.observed_fraction,
        )
