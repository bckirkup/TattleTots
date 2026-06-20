"""Per-user Common Operating Picture: fused situational belief from bot collective inputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tattletots.models.location import EventLocation


def location_key(location: EventLocation) -> str:
    """Serialize a location for dict keys."""
    return f"{location[0]}:{location[1]}"


class LocationBelief(BaseModel):
    """Fused threat belief at a single location for one user."""

    location: EventLocation
    threat_level: float = Field(default=0.0, ge=0.0)
    supporting_reports: int = Field(default=0, ge=0)
    supporting_weight: float = Field(default=0.0, ge=0.0)
    last_response_necessary: bool | None = Field(default=None)
    last_dispatched: bool = Field(default=False)
    unnecessary_dispatch_count: int = Field(default=0, ge=0)


class UserCOP(BaseModel):
    """User-specific common operating picture built from collective bot inputs."""

    user_id: str
    user_name: str = Field(default="")
    time_step: int = Field(default=0, ge=0)
    dispatch_threshold: float = Field(default=1.0, ge=0.0)
    min_supporting_reports: int = Field(default=1, ge=1)
    min_supporting_weight: float = Field(default=0.3, ge=0.0)
    decay_factor: float = Field(default=0.95, gt=0.0, le=1.0)
    beliefs: dict[str, LocationBelief] = Field(default_factory=dict)

    def get_belief(self, location: EventLocation) -> LocationBelief:
        key = location_key(location)
        if key not in self.beliefs:
            self.beliefs[key] = LocationBelief(location=location)
        return self.beliefs[key]

    def decay(self) -> None:
        """Fade stale beliefs each step."""
        for belief in self.beliefs.values():
            belief.threat_level *= self.decay_factor

    def locations_above_threshold(self) -> list[EventLocation]:
        """Return locations whose fused threat exceeds dispatch threshold."""
        return [
            b.location
            for b in self.beliefs.values()
            if b.threat_level >= self.dispatch_threshold
            and b.supporting_reports >= self.min_supporting_reports
            and b.supporting_weight >= self.min_supporting_weight
        ]

    def summary(self) -> dict[str, float | int]:
        """Summary metrics for telemetry."""
        above = self.locations_above_threshold()
        levels = [b.threat_level for b in self.beliefs.values()]
        return {
            "belief_count": len(self.beliefs),
            "locations_above_threshold": len(above),
            "max_threat_level": max(levels) if levels else 0.0,
            "mean_threat_level": sum(levels) / len(levels) if levels else 0.0,
        }
