"""Human user model: attention budget, priorities, and trust state."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


class TrustOutcome(enum.StrEnum):
    """Verified outcome category for asymmetric trust updates."""

    CORRECT_ALARM = "correct_alarm"
    FALSE_ALARM = "false_alarm"
    MISSED_EVENT = "missed_event"
    RESPONSE_NECESSARY = "response_necessary"
    RESPONSE_UNNECESSARY = "response_unnecessary"
    WHISTLEBLOWER_CORROBORATED = "whistleblower_corroborated"
    WHISTLEBLOWER_REFUTED = "whistleblower_refuted"
    ACCUSED_CORROBORATED = "accused_corroborated"


@dataclass(frozen=True)
class TrustUpdateDeltas:
    """Magnitude of trust change per outcome type."""

    pos: float = 0.05
    neg: float = 0.2
    miss: float = 0.1
    response_necessary: float = 0.03
    unnecessary_response: float = 0.15
    whistleblower_corroborated: float = 0.04
    whistleblower_refuted: float = 0.12
    accused_corroborated: float = 0.25


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
        outcome: TrustOutcome,
        *,
        deltas: TrustUpdateDeltas | None = None,
    ) -> None:
        """Update trust based on verified outcomes. Asymmetric: hard to build, easy to destroy."""
        d = deltas or TrustUpdateDeltas()
        current = self.get_trust(agent_id)
        if outcome == TrustOutcome.CORRECT_ALARM:
            new_trust = min(1.0, current + d.pos)
        elif outcome == TrustOutcome.FALSE_ALARM:
            new_trust = max(0.0, current - d.neg)
        elif outcome == TrustOutcome.MISSED_EVENT:
            new_trust = max(0.0, current - d.miss)
        elif outcome == TrustOutcome.RESPONSE_NECESSARY:
            new_trust = min(1.0, current + d.response_necessary)
        elif outcome == TrustOutcome.RESPONSE_UNNECESSARY:
            new_trust = max(0.0, current - d.unnecessary_response)
        elif outcome == TrustOutcome.WHISTLEBLOWER_CORROBORATED:
            new_trust = min(1.0, current + d.whistleblower_corroborated)
        elif outcome == TrustOutcome.WHISTLEBLOWER_REFUTED:
            new_trust = max(0.0, current - d.whistleblower_refuted)
        elif outcome == TrustOutcome.ACCUSED_CORROBORATED:
            new_trust = max(0.0, current - d.accused_corroborated)
        else:
            new_trust = current
        self.trust[agent_id] = new_trust

    def compute_relevance(self, signal_vector: np.ndarray) -> float:
        """Compute relevance of a signal to this user's priorities."""
        from tattletots.engine.relevance import band_relevance

        return band_relevance(self.priority_vector, signal_vector)
