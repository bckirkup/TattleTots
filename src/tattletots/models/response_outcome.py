"""Post-dispatch responder judgment on whether a response was necessary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tattletots.models.location import EventLocation


class ResponseOutcome(BaseModel):
    """Result of executing a physical response and judging its necessity.

    ``response_necessary`` is True when a real problem existed at the target
    location and the response mitigated it at least partially.
    """

    agent_id: str = Field(description="ID of the agent whose report triggered dispatch")
    responder_user_id: str = Field(description="User who judged the response outcome")
    time_step: int = Field(ge=0)
    location: EventLocation = Field(description="Location where response was attempted")
    response_type: str = Field(description="Domain response kind: suppression, spray, patrol")
    dispatched: bool = Field(description="Whether a physical response was executed")
    problem_severity_before: float = Field(ge=0.0)
    problem_severity_after: float = Field(ge=0.0)
    problem_present: bool = Field(description="Whether a problem existed before dispatch")
    mitigated: bool = Field(description="Whether the response partially mitigated the problem")
    response_necessary: bool = Field(
        description="True when problem_present and mitigated (responder affirms necessity)"
    )
