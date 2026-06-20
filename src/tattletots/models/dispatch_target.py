"""Dispatch target selected from a responder's common operating picture."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tattletots.models.location import EventLocation
from tattletots.models.report import Report


class DispatchTarget(BaseModel):
    """A location where the responder COP authorizes physical action."""

    location: EventLocation
    reports: list[Report] = Field(default_factory=list)
    responder_user_id: str
    cop_threat_level: float = Field(default=0.0, ge=0.0)
