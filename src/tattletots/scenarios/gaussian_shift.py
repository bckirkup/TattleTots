"""Built-in smoke test: multivariate Gaussian with distribution shift.

Requirements §8:
- K=10 structured components, Gaussian noise
- Distribution shift at the midpoint
- 2 synthetic users with different priority vectors
- Expected: trophic hierarchy forms, agents specialize, shift triggers
  partial extinction and re-colonization, detection of shift is escalated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.stream import Stream
from tattletots.models.user import User


class GaussianShiftScenario(DomainAdapter):
    """Multivariate Gaussian with K structured components and midpoint shift.

    The data has K=10 independent structured components embedded in D-dimensional
    space, plus isotropic noise. At step `shift_step`, the component structure
    changes (rotation + amplitude change), simulating a regime change that
    agents must detect and adapt to.
    """

    def __init__(
        self,
        n_components: int = 10,
        dimensionality: int = 20,
        noise_std: float = 0.5,
        shift_step: int = 200,
        total_steps: int = 400,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.dimensionality = dimensionality
        self.noise_std = noise_std
        self.shift_step = shift_step
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)

        # Generate structured components (fixed basis vectors)
        self._basis_pre = self.rng.standard_normal((n_components, dimensionality))
        self._amplitudes_pre = self.rng.uniform(1.0, 3.0, size=n_components)

        # Post-shift: rotated basis + changed amplitudes
        rotation = self.rng.standard_normal((dimensionality, dimensionality))
        q, _ = np.linalg.qr(rotation)
        # Rotate each basis vector: (n_components, D) @ (D, D) → (n_components, D)
        self._basis_post = self._basis_pre @ q.T
        self._amplitudes_post = self.rng.uniform(0.5, 2.5, size=n_components)

        self._streams: list[Stream] = []
        self._users: list[User] = []
        self._current_step = 0
        self._setup_streams()
        self._setup_users()

    def _setup_streams(self) -> None:
        """Create raw data streams — one per structured component group."""
        # Split dimensionality into 3 streams as evenly as possible
        n_streams = 3
        base, remainder = divmod(self.dimensionality, n_streams)
        dims = [base + (1 if i < remainder else 0) for i in range(n_streams)]
        for i, d in enumerate(dims):
            stream = Stream(
                stream_type="raw",  # type: ignore[arg-type]
                dimensionality=d,
                label=f"gaussian_stream_{i}",
                current_data=np.zeros(d),
            )
            self._streams.append(stream)

    def _setup_users(self) -> None:
        """Create 2 synthetic users with orthogonal priority vectors."""
        # User A cares about first half of components
        priority_a = np.zeros(self.n_components)
        priority_a[: self.n_components // 2] = 1.0
        priority_a /= np.linalg.norm(priority_a)

        # User B cares about second half
        priority_b = np.zeros(self.n_components)
        priority_b[self.n_components // 2 :] = 1.0
        priority_b /= np.linalg.norm(priority_b)

        self._users = [
            User(
                name="Park Manager Alpha",
                attention_budget=1.0,
                priority_vector=priority_a,
            ),
            User(
                name="Park Manager Beta",
                attention_budget=1.0,
                priority_vector=priority_b,
            ),
        ]

    def get_streams(self) -> list[Stream]:
        return self._streams

    def get_users(self) -> list[User]:
        return self._users

    def step(self, time_step: int) -> None:
        """Generate data for this time step and update streams."""
        self._current_step = time_step
        data = self._generate_data(time_step)

        # Split into stream chunks
        offset = 0
        for stream in self._streams:
            chunk = data[offset : offset + stream.dimensionality]
            stream.update(chunk)
            offset += stream.dimensionality

    def get_ground_truth(self, time_step: int) -> bool:
        """The shift itself is the 'event' — active for a window around shift_step."""
        return abs(time_step - self.shift_step) <= 5

    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        """Domain-specific relevance scoring."""
        return user.compute_relevance(signal_vector)

    def compute_costs(
        self,
        n_escalations: int,
        n_correct: int,
        n_false_alarms: int,
        n_missed: int,
    ) -> dict[str, float]:
        """Simple cost model for the Gaussian scenario."""
        return {
            "surveillance_cost": n_escalations * 0.1,
            "response_cost": n_correct * 1.0 + n_false_alarms * 2.0,
            "damage_cost": n_missed * 5.0,
        }

    def _generate_data(self, time_step: int) -> NDArray[np.float64]:
        """Generate multivariate Gaussian data with structured components."""
        if time_step < self.shift_step:
            basis = self._basis_pre
            amplitudes = self._amplitudes_pre
        else:
            basis = self._basis_post
            amplitudes = self._amplitudes_post

        # Structured signal: sum of component activations
        activations = self.rng.standard_normal(self.n_components) * amplitudes
        structured = activations @ basis

        # Add noise
        noise = self.rng.standard_normal(self.dimensionality) * self.noise_std

        return structured + noise

    def to_config(self) -> dict[str, int | float | str]:
        """Serialize scenario configuration to JSON-compatible dict."""
        return {
            "scenario": "gaussian_shift",
            "n_components": self.n_components,
            "dimensionality": self.dimensionality,
            "noise_std": self.noise_std,
            "shift_step": self.shift_step,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_config(cls, config: dict[str, int | float | str]) -> GaussianShiftScenario:
        """Construct from a configuration dict."""
        return cls(
            n_components=int(config.get("n_components", 10)),
            dimensionality=int(config.get("dimensionality", 20)),
            noise_std=float(config.get("noise_std", 0.5)),
            shift_step=int(config.get("shift_step", 200)),
            total_steps=int(config.get("total_steps", 400)),
        )

    @classmethod
    def from_config_file(cls, path: Path) -> GaussianShiftScenario:
        """Load scenario from a JSON config file."""
        with open(path) as f:
            config = json.load(f)
        return cls.from_config(config)
