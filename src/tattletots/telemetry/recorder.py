"""Telemetry recorder: captures simulation history for analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class TelemetrySummary(TypedDict):
    total_steps: int
    peak_population: int
    final_population: int
    total_births: int
    total_deaths: int
    total_reports: int
    precision: float
    event_prevalence: float
    grounded_yield_share: float
    attention_solvent_fraction: float
    mean_attention_carrying_capacity: float
    initiation_is_degenerate: bool
    initiation_degeneracy_reasons: list[str]
    max_trophic_depth: float
    reached_equilibrium: bool
    total_responses_dispatched: int
    total_responses_judged_necessary: int
    total_responses_judged_unnecessary: int
    responder_necessity_rate: float
    unnecessary_dispatch_rate: float


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
    active_location_count: int = 0
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
    missed_events: int = 0
    mean_working_dim: float = 0.0
    mean_memory_depth: float = 0.0
    n_sensing_strategies: int = 0
    n_residual_policies: int = 0
    responses_dispatched: int = 0
    responses_judged_necessary: int = 0
    responses_judged_unnecessary: int = 0
    n_attention_solvent_agents: int = 0
    n_attention_eligible_agents: int = 0
    attention_carrying_capacity: float = 0.0
    grounded_info_yield: float = 0.0
    ungrounded_info_yield: float = 0.0
    grounded_yield_share: float = 0.0


@dataclass
class TelemetryRecorder:
    """Accumulates step records and provides summary analytics."""

    history: list[StepRecord] = field(default_factory=list)
    initiation_min_grounded_yield_share: float = 0.5
    initiation_attention_insolvency_steps_fraction: float = 0.8
    initiation_min_solvent_fraction: float = 0.5
    initiation_population_capacity_overshoot_factor: float = 1.0

    def configure_initiation_thresholds(
        self,
        *,
        min_grounded_yield_share: float,
        attention_insolvency_steps_fraction: float,
        min_solvent_fraction: float = 0.5,
        population_capacity_overshoot_factor: float = 1.0,
    ) -> None:
        """Set run-level initiation-degeneracy thresholds from simulation config."""
        self.initiation_min_grounded_yield_share = min_grounded_yield_share
        self.initiation_attention_insolvency_steps_fraction = attention_insolvency_steps_fraction
        self.initiation_min_solvent_fraction = min_solvent_fraction
        self.initiation_population_capacity_overshoot_factor = population_capacity_overshoot_factor

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
    def total_grounded_info_yield(self) -> float:
        return sum(r.grounded_info_yield for r in self.history)

    @property
    def total_ungrounded_info_yield(self) -> float:
        return sum(r.ungrounded_info_yield for r in self.history)

    @property
    def total_responses_dispatched(self) -> int:
        return sum(r.responses_dispatched for r in self.history)

    @property
    def total_responses_judged_necessary(self) -> int:
        return sum(r.responses_judged_necessary for r in self.history)

    @property
    def total_responses_judged_unnecessary(self) -> int:
        return sum(r.responses_judged_unnecessary for r in self.history)

    @property
    def max_trophic_depth(self) -> float:
        if not self.history:
            return 0.0
        return max(r.max_trophic_level for r in self.history)

    def population_history(self) -> list[int]:
        """Population count over time."""
        return [r.population for r in self.history]

    def ecology_time_series(self) -> dict[str, list[int] | list[float]]:
        """Ecology metrics over time for unified output schema."""
        return {
            "population": [r.population for r in self.history],
            "reports_issued": [r.reports_issued for r in self.history],
            "correct_reports": [r.correct_reports for r in self.history],
            "false_alarms": [r.false_alarms for r in self.history],
            "missed_events": [r.missed_events for r in self.history],
            "responses_dispatched": [r.responses_dispatched for r in self.history],
            "responses_judged_necessary": [r.responses_judged_necessary for r in self.history],
            "responses_judged_unnecessary": [r.responses_judged_unnecessary for r in self.history],
            "mean_info_energy": [r.mean_info_energy for r in self.history],
            "mean_attn_energy": [r.mean_attn_energy for r in self.history],
            "n_attention_solvent_agents": [r.n_attention_solvent_agents for r in self.history],
            "n_attention_eligible_agents": [r.n_attention_eligible_agents for r in self.history],
            "attention_carrying_capacity": [r.attention_carrying_capacity for r in self.history],
            "grounded_info_yield": [r.grounded_info_yield for r in self.history],
            "ungrounded_info_yield": [r.ungrounded_info_yield for r in self.history],
            "grounded_yield_share": [r.grounded_yield_share for r in self.history],
            "births": [r.births for r in self.history],
            "deaths": [r.deaths for r in self.history],
            "n_compression_types": [r.n_compression_types for r in self.history],
            "max_trophic_level": [r.max_trophic_level for r in self.history],
        }

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
            "grounded_info_yield": [r.grounded_info_yield for r in self.history],
            "ungrounded_info_yield": [r.ungrounded_info_yield for r in self.history],
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

    def _event_prevalence(self) -> float:
        if not self.history:
            return 0.0
        return sum(r.ground_truth_active for r in self.history) / len(self.history)

    def _grounded_yield_share(self) -> float:
        total = self.total_grounded_info_yield + self.total_ungrounded_info_yield
        return self.total_grounded_info_yield / total if total > 0 else 0.0

    def _attention_solvent_fraction(self) -> float:
        fractions = [
            r.n_attention_solvent_agents / r.n_attention_eligible_agents
            for r in self.history
            if r.n_attention_eligible_agents > 0
        ]
        return sum(fractions) / len(fractions) if fractions else 0.0

    def _mean_attention_carrying_capacity(self) -> float:
        capacities = [r.attention_carrying_capacity for r in self.history if r.population > 0]
        return sum(capacities) / len(capacities) if capacities else 0.0

    def initiation_degeneracy(self) -> tuple[bool, list[str]]:
        """Return whether configured initiation degeneracies occurred."""
        if not self.history:
            return True, ["no_telemetry"]

        reasons: list[str] = []
        precision = self.total_correct_reports / max(self.total_reports, 1)
        if precision <= self._event_prevalence():
            reasons.append("precision_not_above_event_prevalence")
        if self._grounded_yield_share() < self.initiation_min_grounded_yield_share:
            reasons.append("grounded_yield_share_below_minimum")

        insolvent_steps = sum(
            r.n_attention_eligible_agents > 0
            and (
                r.n_attention_solvent_agents / r.n_attention_eligible_agents
                < self.initiation_min_solvent_fraction
            )
            for r in self.history
        )
        capacity = self._mean_attention_carrying_capacity()
        capacity_overshoot = (
            capacity > 0
            and self.peak_population
            > capacity * self.initiation_population_capacity_overshoot_factor
        )
        if (
            self.total_births > 0
            and insolvent_steps / len(self.history)
            >= self.initiation_attention_insolvency_steps_fraction
            and capacity_overshoot
        ):
            reasons.append("attention_insolvency_with_capacity_overshoot")

        return bool(reasons), reasons

    def summary(self) -> TelemetrySummary:
        """Summary statistics for the entire run."""
        degenerate, reasons = self.initiation_degeneracy()
        return {
            "total_steps": self.total_steps,
            "peak_population": self.peak_population,
            "final_population": self.history[-1].population if self.history else 0,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_reports": self.total_reports,
            "precision": (self.total_correct_reports / max(self.total_reports, 1)),
            "event_prevalence": self._event_prevalence(),
            "grounded_yield_share": self._grounded_yield_share(),
            "attention_solvent_fraction": self._attention_solvent_fraction(),
            "mean_attention_carrying_capacity": self._mean_attention_carrying_capacity(),
            "initiation_is_degenerate": degenerate,
            "initiation_degeneracy_reasons": reasons,
            "max_trophic_depth": self.max_trophic_depth,
            "reached_equilibrium": self.is_stable(),
            "total_responses_dispatched": self.total_responses_dispatched,
            "total_responses_judged_necessary": self.total_responses_judged_necessary,
            "total_responses_judged_unnecessary": self.total_responses_judged_unnecessary,
            "responder_necessity_rate": (
                self.total_responses_judged_necessary / max(self.total_responses_dispatched, 1)
            ),
            "unnecessary_dispatch_rate": (
                self.total_responses_judged_unnecessary / max(self.total_responses_dispatched, 1)
            ),
        }
