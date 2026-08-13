"""Data stream abstraction: raw and residual multivariate time series."""

from __future__ import annotations

import enum
import uuid

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tattletots.models.observation import ObservationPacket, ObservationStatus, StreamMetadata


class StreamType(enum.StrEnum):
    """Classification of stream origin."""

    RAW = "raw"
    RESIDUAL = "residual"
    OUTPUT = "output"


class Stream(BaseModel):
    """A multivariate time series that agents can consume.

    Streams are either raw (from the domain environment) or residual
    (the unmodeled remainder from an agent's compression).
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream_type: StreamType = Field(default=StreamType.RAW)
    dimensionality: int = Field(ge=1, description="Number of variables in the stream")
    source_agent_id: str | None = Field(
        default=None,
        description="Agent that produced this residual (None for raw streams)",
    )
    current_data: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Current time-step data vector",
    )
    label: str = Field(default="", description="Human-readable label for the stream")
    metadata: StreamMetadata | None = Field(
        default=None,
        description="Optional domain-declared metadata for each feature.",
    )
    current_status: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype="<U8"),
        description="Per-feature observation availability status.",
    )

    @model_validator(mode="after")
    def _validate_metadata(self) -> Stream:
        if self.metadata is not None:
            self.metadata.validate_dimensionality(self.dimensionality)
        if self.current_status.size not in (0, self.dimensionality):
            raise ValueError(
                f"Expected status dimensionality {self.dimensionality}, "
                f"got {self.current_status.size}"
            )
        return self

    def update(
        self,
        data: np.ndarray,
        status: np.ndarray | list[ObservationStatus | str] | None = None,
    ) -> None:
        """Update the stream with new data for the current time step."""
        if data.shape[-1] != self.dimensionality:
            msg = f"Expected dimensionality {self.dimensionality}, got {data.shape[-1]}"
            raise ValueError(msg)
        self.current_data = data
        if status is not None:
            status_array = np.asarray([str(item) for item in status], dtype="<U8")
            if status_array.size != self.dimensionality:
                msg = (
                    f"Expected status dimensionality {self.dimensionality}, got {status_array.size}"
                )
                raise ValueError(msg)
            self.current_status = status_array
        elif self.metadata is not None:
            self.current_status = np.full(
                self.dimensionality,
                ObservationStatus.OBSERVED.value,
                dtype="<U8",
            )

    def observation(self) -> ObservationPacket:
        """Return the current values with their transport metadata."""
        return ObservationPacket.from_stream(
            self.current_data.astype(np.float64, copy=False),
            self.metadata,
            self.current_status,
        )

    @property
    def structured_variance(self) -> float:
        """Estimate of structured (non-noise) variance in current data."""
        if self.current_data.size == 0:
            return 0.0
        return float(np.var(self.current_data))
