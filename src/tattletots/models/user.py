"""Human user model: attention budget, priorities, and trust state."""

from __future__ import annotations

import uuid

import numpy as np
from pydantic import BaseModel, Field


class User(BaseModel):
    """A human stakeholder who allocates cognitive bandwidth to agents.

    Users have finite attention budgets, priority vectors defining what
    they care about, and trust states for each agent updated by verification.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    attention_budget: float = Field(
        default=1.0, gt=0.0, description="Total cognitive bandwidth available per step"
    )
    priority_vector: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="What topics/threats this user cares about (D-dimensional)",
    )
    trust: dict[str, float] = Field(
        default_factory=dict,
        description="Trust state per agent_id, values in [0, 1]",
    )

    def get_trust(self, agent_id: str) -> float:
        """Get trust for an agent, defaulting to 0.5 for unknown agents."""
        return self.trust.get(agent_id, 0.5)

    def update_trust(
        self,
        agent_id: str,
        *,
        correct_alarm: bool = False,
        false_alarm: bool = False,
        missed_event: bool = False,
        response_necessary: bool = False,
        response_unnecessary: bool = False,
        whistleblower_corroborated: bool = False,
        whistleblower_refuted: bool = False,
        accused_corroborated: bool = False,
        delta_pos: float = 0.05,
        delta_neg: float = 0.2,
        delta_miss: float = 0.1,
        delta_response_necessary: float = 0.03,
        delta_unnecessary_response: float = 0.15,
        delta_whistleblower_corroborated: float = 0.04,
        delta_whistleblower_refuted: float = 0.12,
        delta_accused_corroborated: float = 0.25,
    ) -> None:
        """Update trust based on verified outcomes. Asymmetric: hard to build, easy to destroy."""
        current = self.get_trust(agent_id)
        if correct_alarm:
            new_trust = min(1.0, current + delta_pos)
        elif false_alarm:
            new_trust = max(0.0, current - delta_neg)
        elif missed_event:
            new_trust = max(0.0, current - delta_miss)
        elif response_necessary:
            new_trust = min(1.0, current + delta_response_necessary)
        elif response_unnecessary:
            new_trust = max(0.0, current - delta_unnecessary_response)
        elif whistleblower_corroborated:
            new_trust = min(1.0, current + delta_whistleblower_corroborated)
        elif whistleblower_refuted:
            new_trust = max(0.0, current - delta_whistleblower_refuted)
        elif accused_corroborated:
            new_trust = max(0.0, current - delta_accused_corroborated)
        else:
            new_trust = current
        self.trust[agent_id] = new_trust

    def compute_relevance(self, signal_vector: np.ndarray) -> float:
        """Compute relevance of a signal to this user's priorities."""
        if self.priority_vector.size == 0 or signal_vector.size == 0:
            return 0.0
        if self.priority_vector.shape != signal_vector.shape:
            min_dim = min(len(self.priority_vector), len(signal_vector))
            return float(np.dot(self.priority_vector[:min_dim], signal_vector[:min_dim]))
        return float(np.dot(self.priority_vector, signal_vector))
