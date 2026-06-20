"""Whistleblower report: explicit false-report suspicion from one Tot to a user."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tattletots.models.location import EventLocation


class WhistleblowerReport(BaseModel):
    """Structured 'I suspect a false report' signal."""

    whistleblower_id: str
    accused_agent_id: str
    target_user_id: str
    time_step: int = Field(ge=0)
    location: EventLocation
    suspicion_score: float = Field(ge=0.0)
    basis: str = Field(
        description="unnecessary_response"
    )
