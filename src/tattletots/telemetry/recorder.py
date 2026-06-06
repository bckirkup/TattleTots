"""Telemetry recorder: captures simulation history for analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepRecord:
    """Telemetry snapshot for a single simulation step."""

    time_step: int
    population: int
    births: int
    deaths: int
    reports_issued: int
    correct_reports: int
    false_alarms: int
    mean_info_energy: float
    mean_attn_energy: float
    max_trophic_level: float
    n_streams: int
    ground_truth_active: bool
    # Energy flow tracking
    total_info_yield: float = 0.0
    total_attn_income: float = 0.0
    total_compute_cost: float = 0.0
    total_maintenance_cost: float = 0.0
    # Demographic tracking
    n_juveniles: int = 0
    n_adults: int = 0
    mean_generation: float = 0.0
    n_compression_types: int = 0


@dataclass
class TelemetryRecorder:
    """Accumulates step records and provides summary analytics."""

    history: list[StepRecord] = field(default_factory=list)

    def record_step(self, record: StepRecord) -> None:
        """Append a step record."""
        self.history.append(record)

    @property
    def total_steps(self) -> int:
        return len(self.history)

    @property
    def peak_population(self) -> int:
        if not self.history:
            return 0
        return max(r.population for r in self.history)

    @property
    def total_births(self) -> int:
        return sum(r.births for r in self.history)

    @property
    def total_deaths(self) -> int:
        return sum(r.deaths for r in self.history)

    @property
    def total_reports(self) -> int:
        return sum(r.reports_issued for r in self.history)

    @property
    def total_correct_reports(self) -> int:
        return sum(r.correct_reports for r in self.history)

    @property
    def total_false_alarms(self) -> int:
        return sum(r.false_alarms for r in self.history)

    @property
    def max_trophic_depth(self) -> float:
        if not self.history:
            return 0.0
        return max(r.max_trophic_level for r in self.history)

    def population_history(self) -> list[int]:
        """Population count over time."""
        return [r.population for r in self.history]

    def is_stable(self, window: int = 50, tolerance: float = 0.2) -> bool:
        """Check if population has reached approximate equilibrium.

        Stable = variance in last `window` steps is within `tolerance` of mean.
        """
        if len(self.history) < window:
            return False
        recent = [r.population for r in self.history[-window:]]
        mean_pop = sum(recent) / len(recent)
        if mean_pop == 0:
            return False
        variance = sum((p - mean_pop) ** 2 for p in recent) / len(recent)
        cv = (variance**0.5) / mean_pop
        return bool(cv < tolerance)

    def extinction_cascade_detected(self) -> bool:
        """Check if a sudden population crash occurred (>50% in 10 steps)."""
        if len(self.history) < 10:
            return False
        for i in range(10, len(self.history)):
            before = self.history[i - 10].population
            after = self.history[i].population
            if before > 0 and after / before < 0.5:
                return True
        return False

    def energy_flow_history(self) -> dict[str, list[float]]:
        """Energy flow metrics over time."""
        return {
            "info_yield": [r.total_info_yield for r in self.history],
            "attn_income": [r.total_attn_income for r in self.history],
            "compute_cost": [r.total_compute_cost for r in self.history],
            "maintenance_cost": [r.total_maintenance_cost for r in self.history],
        }

    def demographic_history(self) -> dict[str, list[float]]:
        """Demographic metrics over time."""
        return {
            "juveniles": [float(r.n_juveniles) for r in self.history],
            "adults": [float(r.n_adults) for r in self.history],
            "mean_generation": [r.mean_generation for r in self.history],
            "compression_types": [float(r.n_compression_types) for r in self.history],
        }

    def summary(self) -> dict[str, object]:
        """Summary statistics for the entire run."""
        return {
            "total_steps": self.total_steps,
            "peak_population": self.peak_population,
            "final_population": self.history[-1].population if self.history else 0,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_reports": self.total_reports,
            "precision": (self.total_correct_reports / max(self.total_reports, 1)),
            "max_trophic_depth": self.max_trophic_depth,
            "reached_equilibrium": self.is_stable(),
        }
