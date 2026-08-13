"""Read-only reporter-policy interface for ordinary-economy simulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from tattletots.models.location import EventLocation, LocationFrame
from tattletots.models.observation import StreamMetadata
from tattletots.models.stream import Stream


@dataclass(frozen=True)
class ReporterMetadata:
    """Immutable snapshot of published stream declarations."""

    coordinates: tuple[tuple[float, ...] | None, ...] | None = None
    sensor_coordinates: tuple[tuple[float, ...] | None, ...] | None = None
    modality: tuple[str | None, ...] | None = None
    identity: tuple[str | None, ...] | None = None
    footprints: tuple[tuple[float, ...] | None, ...] | None = None
    resolution: tuple[float | None, ...] | None = None

    @classmethod
    def from_stream_metadata(cls, metadata: StreamMetadata | None) -> ReporterMetadata:
        if metadata is None:
            return cls()
        return cls(
            coordinates=_as_tuple(metadata.coordinates),
            sensor_coordinates=_as_tuple(metadata.sensor_coordinates),
            modality=_as_tuple(metadata.modality),
            identity=_as_tuple(metadata.identity),
            footprints=_as_tuple(metadata.footprints),
            resolution=_as_tuple(metadata.resolution),
        )


@dataclass(frozen=True)
class ReporterStream:
    """Immutable snapshot of one raw stream consumed by a reporter."""

    label: str
    data: NDArray[np.float64]
    observation_status: tuple[str, ...]
    metadata: ReporterMetadata

    @classmethod
    def from_stream(cls, stream: Stream) -> ReporterStream:
        data = stream.current_data.astype(np.float64, copy=True)
        data.setflags(write=False)
        return cls(
            label=stream.label,
            data=data,
            observation_status=tuple(str(item) for item in stream.current_status),
            metadata=ReporterMetadata.from_stream_metadata(stream.metadata),
        )


@dataclass(frozen=True)
class ReporterPolicyContext:
    """Only the observations and public declarations available to a reporter."""

    observation: NDArray[np.float64]
    projected_input: NDArray[np.float64]
    signal_vector: NDArray[np.float64]
    anomaly_score: float
    escalation_threshold: float
    time_step: int
    location_frame: LocationFrame | None
    streams: tuple[ReporterStream, ...]


@dataclass(frozen=True)
class ReporterDecision:
    """The two decisions made by a reporter policy for one step."""

    escalate: bool
    location: EventLocation | None = None


class ReporterPolicy(Protocol):
    """Protocol implemented by a declaratively selected reporter policy."""

    def decide(self, context: ReporterPolicyContext) -> ReporterDecision:
        """Decide whether to report and which public location to name."""


ReporterPolicyFactory = Callable[[], ReporterPolicy]
_REPORTER_POLICY_FACTORIES: dict[str, ReporterPolicyFactory] = {}
_MetadataItem = TypeVar("_MetadataItem")


def register_reporter_policy(name: str, factory: ReporterPolicyFactory) -> None:
    """Register a named reporter-policy factory."""
    if not name:
        raise ValueError("reporter policy name must not be empty")
    _REPORTER_POLICY_FACTORIES[name] = factory


def create_reporter_policy(name: str) -> ReporterPolicy:
    """Create a named reporter policy, failing loudly for unknown names."""
    try:
        factory = _REPORTER_POLICY_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown reporter policy {name!r}") from exc
    return factory()


def _as_tuple(
    values: list[_MetadataItem] | None,
) -> tuple[_MetadataItem, ...] | None:
    return None if values is None else tuple(values)
