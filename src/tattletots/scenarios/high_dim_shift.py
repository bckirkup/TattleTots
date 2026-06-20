"""High-dimensional smoke test: 1000-dim stream with localized block shift.

Agents must evolve block specialization and sensing strategies to detect
a shift confined to one spatial block.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.location import EventLocation
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User


class HighDimShiftScenario(DomainAdapter):
    """Single high-dimensional stream with localized distribution shift."""

    def __init__(
        self,
        dimensionality: int = 1000,
        n_blocks: int = 10,
        shift_block: int = 3,
        noise_std: float = 0.5,
        shift_step: int = 200,
        total_steps: int = 400,
        seed: int = 42,
    ) -> None:
        self.dimensionality = dimensionality
        self.n_blocks = n_blocks
        self.shift_block = shift_block
        self.noise_std = noise_std
        self.shift_step = shift_step
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        self.block_size = dimensionality // n_blocks

        self._streams: list[Stream] = []
        self._users: list[User] = []
        self._current_step = 0
        self._setup_streams()
        self._setup_users()

    def _setup_streams(self) -> None:
        stream = Stream(
            stream_type=StreamType.RAW,
            dimensionality=self.dimensionality,
            label="high_dim_stream",
            current_data=np.zeros(self.dimensionality),
        )
        self._streams.append(stream)

    def _setup_users(self) -> None:
        priority = np.zeros(self.n_blocks)
        priority[self.shift_block] = 1.0
        norm = float(np.linalg.norm(priority))
        priority /= max(norm, 1e-10)
        self._users = [
            User(
                name="Block Monitor",
                attention_budget=1.0,
                priority_vector=priority,
            ),
        ]

    def get_streams(self) -> list[Stream]:
        return self._streams

    def get_users(self) -> list[User]:
        return self._users

    def step(self, time_step: int) -> None:
        self._current_step = time_step
        data = self._generate_data(time_step)
        self._streams[0].update(data)

    def get_ground_truth(self, time_step: int) -> bool:
        return abs(time_step - self.shift_step) <= 5

    def get_active_locations(self, time_step: int) -> list[EventLocation]:
        if not self.get_ground_truth(time_step):
            return []
        return [(self.shift_block, 0)]

    def infer_report_location(
        self,
        stream_data: list[NDArray[np.float64]],
        stream_labels: list[str],
    ) -> EventLocation:
        if not stream_data:
            return (0, 0)
        data = stream_data[0]
        peak_idx = int(np.argmax(np.abs(data)))
        return self.dim_index_to_location(peak_idx)

    def dim_index_to_location(self, dim_index: int) -> EventLocation:
        block = min(dim_index // self.block_size, self.n_blocks - 1)
        return (block, 0)

    def get_spatial_dim_map(self) -> dict[str, slice]:
        return {
            "high_dim_stream": slice(0, self.dimensionality),
        }

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
        targets: list,
        time_step: int,
    ) -> list[ResponseOutcome]:
        """Built-in scenario has no physical response actions."""
        return []

    def _generate_data(self, time_step: int) -> NDArray[np.float64]:
        data = self.rng.standard_normal(self.dimensionality) * self.noise_std
        # Baseline structure in all blocks
        for b in range(self.n_blocks):
            start = b * self.block_size
            end = start + self.block_size
            data[start:end] += self.rng.standard_normal(self.block_size) * 0.3

        if time_step >= self.shift_step:
            start = self.shift_block * self.block_size
            end = start + self.block_size
            data[start:end] += self.rng.standard_normal(self.block_size) * 2.0

        return data

    def get_ground_truth_vector(self, time_step: int) -> NDArray[np.float64]:
        """Reference vector for whistleblowing (expected signal)."""
        data = np.zeros(self.dimensionality)
        if time_step >= self.shift_step:
            start = self.shift_block * self.block_size
            end = start + self.block_size
            data[start:end] = 1.0
        return data

    def to_config(self) -> dict[str, int | float | str]:
        return {
            "scenario": "high_dim_shift",
            "dimensionality": self.dimensionality,
            "n_blocks": self.n_blocks,
            "shift_block": self.shift_block,
            "noise_std": self.noise_std,
            "shift_step": self.shift_step,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_config(cls, config: dict[str, int | float | str]) -> HighDimShiftScenario:
        return cls(
            dimensionality=int(config.get("dimensionality", 1000)),
            n_blocks=int(config.get("n_blocks", 10)),
            shift_block=int(config.get("shift_block", 3)),
            noise_std=float(config.get("noise_std", 0.5)),
            shift_step=int(config.get("shift_step", 200)),
            total_steps=int(config.get("total_steps", 400)),
        )

    @classmethod
    def from_config_file(cls, path: Path) -> HighDimShiftScenario:
        with open(path) as f:
            config = json.load(f)
        return cls.from_config(config)
