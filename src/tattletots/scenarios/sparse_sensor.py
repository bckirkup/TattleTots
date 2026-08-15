"""Sparse, noisy, intermittent point-sensor localization scenario."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.dispatch_target import DispatchTarget
from tattletots.models.location import EventLocation
from tattletots.models.observation import ObservationStatus, StreamMetadata
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User

MAX_GRID_SIZE = 512
MAX_SENSOR_COUNT = MAX_GRID_SIZE**2
MAX_TOTAL_STEPS = 100_000


class SparseSensorScenario(DomainAdapter):
    """A possible but noisy localization problem over a published sensor grid."""

    def __init__(
        self,
        grid_size: int = 10,
        n_sensors: int = 24,
        noise_std: float = 0.2,
        dropout_rate: float = 0.1,
        decay_length: float = 3.0,
        total_steps: int = 200,
        seed: int = 42,
    ) -> None:
        self._validate_parameters(
            grid_size=grid_size,
            n_sensors=n_sensors,
            noise_std=noise_std,
            dropout_rate=dropout_rate,
            decay_length=decay_length,
            total_steps=total_steps,
        )
        self.grid_size = grid_size
        self.n_sensors = n_sensors
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.decay_length = decay_length
        self.total_steps = total_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Keep sink bounds visible to static taint analysis; validation makes these no-ops.
        bounded_grid_size = min(grid_size, MAX_GRID_SIZE)
        bounded_sensor_count = min(n_sensors, MAX_SENSOR_COUNT)
        bounded_total_steps = min(total_steps, MAX_TOTAL_STEPS)
        all_locations = np.array(
            [(row, col) for row in range(bounded_grid_size) for col in range(bounded_grid_size)],
            dtype=np.int64,
        )
        selected = self.rng.choice(len(all_locations), size=bounded_sensor_count, replace=False)
        self.sensor_locations = all_locations[selected]
        self._source_locations = self.rng.integers(
            0, bounded_grid_size, size=(bounded_total_steps, 2), dtype=np.int64
        )
        self._streams: list[Stream] = []
        self._users: list[User] = []
        self._setup_streams()
        self._setup_users()

    @staticmethod
    def _validate_parameters(
        *,
        grid_size: int,
        n_sensors: int,
        noise_std: float,
        dropout_rate: float,
        decay_length: float,
        total_steps: int,
    ) -> None:
        if not 1 <= grid_size <= MAX_GRID_SIZE:
            raise ValueError(f"grid_size must be between 1 and {MAX_GRID_SIZE}")
        if not 1 <= total_steps <= MAX_TOTAL_STEPS:
            raise ValueError(f"total_steps must be between 1 and {MAX_TOTAL_STEPS}")
        if not 1 <= n_sensors <= grid_size**2:
            raise ValueError("n_sensors must be at least 1 and no greater than grid_size squared")
        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        if not math.isfinite(dropout_rate) or not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be finite and between 0 and 1")
        if not math.isfinite(decay_length) or decay_length <= 0.0:
            raise ValueError("decay_length must be finite and strictly positive")

    def _setup_streams(self) -> None:
        bounded_sensor_count = min(self.n_sensors, MAX_SENSOR_COUNT)
        coordinates: list[tuple[float, ...] | None] = [
            tuple(map(float, location)) for location in self.sensor_locations
        ]
        metadata = StreamMetadata(
            sensor_coordinates=coordinates,
            modality=["point_sensor"] * bounded_sensor_count,
            resolution=[1.0] * bounded_sensor_count,
        )
        self._streams = [
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=bounded_sensor_count,
                label="sparse_point_sensors",
                current_data=np.zeros(bounded_sensor_count, dtype=np.float64),
                metadata=metadata,
            )
        ]

    def _setup_users(self) -> None:
        self._users = [
            User(
                name="Sparse Sensor Monitor",
                attention_budget=1.0,
                priority_vector=np.ones(1, dtype=np.float64),
            )
        ]

    def get_streams(self) -> list[Stream]:
        return self._streams

    def get_users(self) -> list[User]:
        return self._users

    def step(self, time_step: int) -> None:
        source = self._source_locations[time_step % self.total_steps]
        distances = np.linalg.norm(self.sensor_locations - source, axis=1)
        signal = np.exp(-distances / self.decay_length)
        data = signal + self.rng.normal(0.0, self.noise_std, size=self.n_sensors)
        missing = self.rng.random(self.n_sensors) < self.dropout_rate
        data[missing] = 0.0
        status = np.full(self.n_sensors, ObservationStatus.OBSERVED.value, dtype="<U8")
        status[missing] = ObservationStatus.MISSING.value
        self._streams[0].update(data, status=status)

    def get_ground_truth(self, time_step: int) -> bool:
        return 0 <= time_step < self.total_steps

    def get_active_locations(self, time_step: int) -> list[EventLocation]:
        source = self._source_locations[time_step % self.total_steps]
        return [(int(source[0]), int(source[1]))]

    def infer_report_location(
        self,
        stream_data: list[NDArray[np.float64]],
        stream_labels: list[str],
    ) -> EventLocation:
        """Legacy callback baseline: choose the strongest published sensor."""
        matching = [
            data
            for data, label in zip(stream_data, stream_labels, strict=True)
            if label == "sparse_point_sensors" and data.size == self.n_sensors
        ]
        if not matching:
            return (0, 0)
        index = int(np.argmax(np.abs(matching[0])))
        location = self.sensor_locations[index]
        return (int(location[0]), int(location[1]))

    def reference_max_signal(self) -> EventLocation:
        """Simple baseline: report the coordinate of the strongest observed sensor."""
        data = self._streams[0].current_data
        status = self._streams[0].current_status
        observed = status == ObservationStatus.OBSERVED.value
        index = int(np.argmax(np.where(observed, data, -np.inf)))
        location = self.sensor_locations[index]
        return (int(location[0]), int(location[1]))

    def reference_weighted_centroid(self) -> EventLocation:
        """Simple baseline: round the positive-signal weighted sensor centroid."""
        data = self._streams[0].current_data
        status = self._streams[0].current_status
        observed = status == ObservationStatus.OBSERVED.value
        weights = np.maximum(np.where(observed, data, 0.0), 0.0)
        if float(weights.sum()) <= 0.0:
            return self.reference_max_signal()
        centroid = np.average(self.sensor_locations, axis=0, weights=weights)
        return (int(round(centroid[0])), int(round(centroid[1])))

    def reference_inverse_distance(self) -> EventLocation:
        """Simple baseline: interpolate positive signal over the published grid."""
        data = self._streams[0].current_data
        status = self._streams[0].current_status
        observed = status == ObservationStatus.OBSERVED.value
        candidates = np.array(
            [(row, col) for row in range(self.grid_size) for col in range(self.grid_size)],
            dtype=np.float64,
        )
        distances = np.linalg.norm(
            candidates[:, None, :] - self.sensor_locations[None, :, :],
            axis=2,
        )
        weights = np.where(observed, 1.0 / (1.0 + distances) ** 2, 0.0)
        values = np.maximum(data, 0.0)
        scores = (weights * values[None, :]).sum(axis=1) / (weights.sum(axis=1) + 1e-12)
        location = candidates[int(np.argmax(scores))]
        return (int(location[0]), int(location[1]))

    def achievable_location_bound(self) -> EventLocation:
        """Use the published signal model as an empirical available-evidence bound."""
        data = self._streams[0].current_data
        status = self._streams[0].current_status
        observed = status == ObservationStatus.OBSERVED.value
        candidates = np.array(
            [(row, col) for row in range(self.grid_size) for col in range(self.grid_size)],
            dtype=np.float64,
        )
        distances = np.linalg.norm(
            self.sensor_locations[:, None, :] - candidates[None, :, :],
            axis=2,
        )
        profile = np.exp(-distances / self.decay_length)
        values = data[observed]
        profiles = profile[observed]
        amplitudes = (profiles * values[:, None]).sum(axis=0) / (
            (profiles * profiles).sum(axis=0) + 1e-12
        )
        errors = ((values[:, None] - amplitudes[None, :] * profiles) ** 2).mean(axis=0)
        location = candidates[int(np.argmin(errors))]
        return (int(location[0]), int(location[1]))

    def dim_index_to_location(self, dim_index: int) -> EventLocation:
        location = self.sensor_locations[dim_index % self.n_sensors]
        return (int(location[0]), int(location[1]))

    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        from tattletots.engine.relevance import score_report_relevance

        return score_report_relevance(signal_vector, user)

    def compute_costs(
        self,
        n_escalations: int,
        n_correct: int,
        n_false_alarms: int,
        n_missed: int,
    ) -> dict[str, float]:
        return {
            "surveillance_cost": n_escalations * 0.1,
            "response_cost": n_correct * 1.0 + n_false_alarms * 2.0,
            "damage_cost": n_missed * 5.0,
        }

    def get_responder_user_id(self) -> str:
        return self._users[0].id

    def dispatch_and_judge_responses(
        self,
        _targets: list[DispatchTarget],
        _time_step: int,
    ) -> list[ResponseOutcome]:
        return []

    def to_config(self) -> dict[str, int | float | str]:
        return {
            "scenario": "sparse_sensor",
            "grid_size": self.grid_size,
            "n_sensors": self.n_sensors,
            "noise_std": self.noise_std,
            "dropout_rate": self.dropout_rate,
            "decay_length": self.decay_length,
            "total_steps": self.total_steps,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, int | float | str]) -> SparseSensorScenario:
        return cls(
            grid_size=int(config.get("grid_size", 10)),
            n_sensors=int(config.get("n_sensors", 24)),
            noise_std=float(config.get("noise_std", 0.2)),
            dropout_rate=float(config.get("dropout_rate", 0.1)),
            decay_length=float(config.get("decay_length", 3.0)),
            total_steps=int(config.get("total_steps", 200)),
            seed=int(config.get("seed", 42)),
        )

    @classmethod
    def from_config_file(cls, path: Path) -> SparseSensorScenario:
        with open(path) as handle:
            return cls.from_config(json.load(handle))
