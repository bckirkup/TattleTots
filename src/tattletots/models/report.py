"""Escalation report: what agents communicate to users."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from tattletots.models.location import EventLocation


class Report(BaseModel):
    """An escalation event: an agent reporting a detected anomaly to a user.

    Reports carry the signal vector, confidence, and are evaluated against
    ground truth to update trust.
    """

    model_config = {"arbitrary_types_allowed": True}

    agent_id: str = Field(description="ID of the reporting agent")
    target_user_id: str = Field(description="ID of the user this report is directed to")
    time_step: int = Field(ge=0, description="When the report was generated")
    signal_vector: np.ndarray = Field(description="The compressed signal being escalated")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence in this anomaly")
    anomaly_score: float = Field(ge=0.0, description="Raw anomaly score that triggered escalation")
    location: EventLocation = Field(
        description="Reported location of the detected event (domain-specific coordinates)"
    )
    verified: bool = Field(default=False, description="Whether this report has been verified")
    correct: bool | None = Field(
        default=None, description="Whether the report was correct (None if unverified)"
    )
