"""Abstract base class for domain adapters.

Domain repositories (FireEcology, CruiseEcology, etc.) implement this
interface to plug into the TattleTots engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from tattletots.models.stream import Stream
from tattletots.models.user import User


class DomainAdapter(ABC):
    """Contract between TattleTots and a domain simulation.

    The domain is responsible for:
    1. Generating data streams each time step
    2. Providing ground truth (what actually happened)
    3. Defining user profiles
    4. Scoring relevance
    5. Computing domain costs
    """

    @abstractmethod
    def get_streams(self) -> list[Stream]:
        """Return the domain's data streams (initialized, ready for first step)."""

    @abstractmethod
    def get_users(self) -> list[User]:
        """Return the domain's user profiles."""

    @abstractmethod
    def step(self, time_step: int) -> None:
        """Advance the domain simulation by one step.

        After this call, streams should have updated current_data.
        """

    @abstractmethod
    def get_ground_truth(self, time_step: int) -> bool:
        """Return whether a true event is active at this time step."""

    @abstractmethod
    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        """Score how relevant a signal is to a specific user (domain-specific)."""

    @abstractmethod
    def compute_costs(
        self,
        n_escalations: int,
        n_correct: int,
        n_false_alarms: int,
        n_missed: int,
    ) -> dict[str, float]:
        """Compute domain-specific costs.

        Returns dict with keys: surveillance_cost, response_cost, damage_cost.
        """
