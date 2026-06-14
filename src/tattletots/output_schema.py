"""Unified output schema for TattleTots simulation runs.

All domain adapters produce results conforming to this schema,
enabling cross-domain comparison and analysis.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class RunSummary(BaseModel):
    """High-level run metadata."""

    domain: str = Field(description="Domain identifier (e.g. 'fire_ecology', 'coral_key')")
    steps_completed: int = Field(ge=0)
    seed: int | None = Field(default=None)
    wall_time_seconds: float = Field(default=0.0, ge=0.0)
    tattletots_version: str = Field(default="0.1.0")


class EcologyMetrics(BaseModel):
    """Metrics from the TattleTots agent ecology."""

    final_population: int = Field(default=0, ge=0)
    peak_population: int = Field(default=0, ge=0)
    total_births: int = Field(default=0, ge=0)
    total_deaths: int = Field(default=0, ge=0)
    total_reports: int = Field(default=0, ge=0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    max_trophic_depth: float = Field(default=0.0, ge=0.0)
    reached_equilibrium: bool = Field(default=False)


class CostMetrics(BaseModel):
    """Cost accounting from the domain adapter."""

    total_surveillance_cost: float = Field(default=0.0, ge=0.0)
    total_response_cost: float = Field(default=0.0, ge=0.0)
    total_damage_cost: float = Field(default=0.0, ge=0.0)
    total_cost: float = Field(default=0.0, ge=0.0)
    mean_cost_per_step: float = Field(default=0.0, ge=0.0)


class TimeSeries(BaseModel):
    """Time-series data for post-hoc analysis."""

    population: list[int] = Field(default_factory=list)
    cost_per_step: list[float] = Field(default_factory=list)


class SimulationOutput(BaseModel):
    """Unified output schema for all TattleTots-integrated simulations.

    Provides a consistent structure for cross-domain analysis while
    allowing domain-specific metrics via the `domain_metrics` field.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run_summary: RunSummary
    simulation_config: dict[str, Any] = Field(
        default_factory=dict, description="TattleTots engine configuration"
    )
    domain_config: dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific configuration"
    )
    ecology_metrics: EcologyMetrics = Field(default_factory=EcologyMetrics)
    cost_metrics: CostMetrics = Field(default_factory=CostMetrics)
    domain_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific metrics (varies per adapter)"
    )
    time_series: TimeSeries = Field(default_factory=TimeSeries)

    def write_json(self, path: Path | str) -> None:
        """Write results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def read_json(cls, path: Path | str) -> SimulationOutput:
        """Read results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
