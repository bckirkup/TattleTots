"""Data stream abstraction: raw and residual multivariate time series."""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


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

    def update(self, data: np.ndarray) -> None:
        """Update the stream with new data for the current time step."""
        if data.shape[-1] != self.dimensionality:
            msg = f"Expected dimensionality {self.dimensionality}, got {data.shape[-1]}"
            raise ValueError(msg)
        self.current_data = data

    @property
    def structured_variance(self) -> float:
        """Estimate of structured (non-noise) variance in current data."""
        if self.current_data.size == 0:
            return 0.0
        return float(np.var(self.current_data))
