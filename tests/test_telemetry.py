"""Unit tests for telemetry/recorder.py and telemetry/cost_accounting.py."""

from __future__ import annotations

from tattletots.telemetry.cost_accounting import CostAccumulator, StepCosts
from tattletots.telemetry.recorder import StepRecord, TelemetryRecorder


def _make_record(
    time_step: int = 1,
    population: int = 10,
    births: int = 0,
    deaths: int = 0,
    reports_issued: int = 0,
    correct_reports: int = 0,
    false_alarms: int = 0,
    mean_info_energy: float = 1.0,
    mean_attn_energy: float = 1.0,
    max_trophic_level: float = 1.0,
    n_streams: int = 3,
    ground_truth_active: bool = False,
    **kwargs: float | int,
) -> StepRecord:
    return StepRecord(
        time_step=time_step,
        population=population,
        births=births,
        deaths=deaths,
        reports_issued=reports_issued,
        correct_reports=correct_reports,
        false_alarms=false_alarms,
        mean_info_energy=mean_info_energy,
        mean_attn_energy=mean_attn_energy,
        max_trophic_level=max_trophic_level,
        n_streams=n_streams,
        ground_truth_active=ground_truth_active,
        **kwargs,
    )


class TestTelemetryRecorder:
    def test_empty_recorder(self) -> None:
        rec = TelemetryRecorder()
        assert rec.total_steps == 0
        assert rec.peak_population == 0
        assert rec.total_births == 0
        assert rec.total_deaths == 0
        assert rec.total_reports == 0
        assert rec.max_trophic_depth == 0.0
        assert not rec.is_stable()
        assert not rec.extinction_cascade_detected()
        assert rec.population_history() == []

    def test_record_and_query(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(_make_record(time_step=1, population=10, births=2, deaths=1))
        rec.record_step(_make_record(time_step=2, population=11, births=1, deaths=0))
        assert rec.total_steps == 2
        assert rec.peak_population == 11
        assert rec.total_births == 3
        assert rec.total_deaths == 1
        assert rec.population_history() == [10, 11]

    def test_is_stable_flat_population(self) -> None:
        rec = TelemetryRecorder()
        for i in range(60):
            rec.record_step(_make_record(time_step=i, population=20))
        assert rec.is_stable(window=50, tolerance=0.2)

    def test_is_stable_false_for_volatile(self) -> None:
        rec = TelemetryRecorder()
        for i in range(60):
            pop = 5 if i % 2 == 0 else 50
            rec.record_step(_make_record(time_step=i, population=pop))
        assert not rec.is_stable(window=50, tolerance=0.2)

    def test_is_stable_false_when_too_few_steps(self) -> None:
        rec = TelemetryRecorder()
        for i in range(10):
            rec.record_step(_make_record(time_step=i, population=20))
        assert not rec.is_stable(window=50)

    def test_is_stable_false_when_zero_population(self) -> None:
        rec = TelemetryRecorder()
        for i in range(60):
            rec.record_step(_make_record(time_step=i, population=0))
        assert not rec.is_stable(window=50)

    def test_extinction_cascade_detected(self) -> None:
        rec = TelemetryRecorder()
        # 10 steps at pop=100, then sudden drop to 10
        for i in range(10):
            rec.record_step(_make_record(time_step=i, population=100))
        rec.record_step(_make_record(time_step=10, population=10))
        assert rec.extinction_cascade_detected()

    def test_no_extinction_cascade_gradual_decline(self) -> None:
        rec = TelemetryRecorder()
        for i in range(20):
            rec.record_step(_make_record(time_step=i, population=max(1, 100 - i * 3)))
        assert not rec.extinction_cascade_detected()

    def test_max_trophic_depth(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(_make_record(time_step=1, max_trophic_level=2.0))
        rec.record_step(_make_record(time_step=2, max_trophic_level=3.5))
        rec.record_step(_make_record(time_step=3, max_trophic_level=2.5))
        assert rec.max_trophic_depth == 3.5

    def test_total_reports_and_precision(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(time_step=1, reports_issued=10, correct_reports=7, false_alarms=3)
        )
        rec.record_step(
            _make_record(time_step=2, reports_issued=5, correct_reports=5, false_alarms=0)
        )
        assert rec.total_reports == 15
        assert rec.total_correct_reports == 12
        assert rec.total_false_alarms == 3

    def test_summary_returns_all_keys(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(_make_record(time_step=1))
        s = rec.summary()
        expected_keys = {
            "total_steps",
            "peak_population",
            "final_population",
            "total_births",
            "total_deaths",
            "total_reports",
            "precision",
            "max_trophic_depth",
            "reached_equilibrium",
        }
        assert set(s.keys()) == expected_keys

    def test_energy_flow_history(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                time_step=1,
                total_info_yield=5.0,
                total_attn_income=3.0,
                total_compute_cost=1.0,
                total_maintenance_cost=0.5,
            )
        )
        h = rec.energy_flow_history()
        assert h["info_yield"] == [5.0]
        assert h["attn_income"] == [3.0]
        assert h["compute_cost"] == [1.0]
        assert h["maintenance_cost"] == [0.5]

    def test_demographic_history(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                time_step=1,
                n_juveniles=3,
                n_adults=7,
                mean_generation=2.5,
                n_compression_types=3,
            )
        )
        d = rec.demographic_history()
        assert d["juveniles"] == [3.0]
        assert d["adults"] == [7.0]
        assert d["mean_generation"] == [2.5]
        assert d["compression_types"] == [3.0]


class TestCostAccumulator:
    def test_empty_accumulator(self) -> None:
        acc = CostAccumulator()
        assert acc.total_surveillance == 0.0
        assert acc.total_response == 0.0
        assert acc.total_damage == 0.0
        assert acc.total_cost == 0.0
        assert acc.mean_cost_per_step() == 0.0
        assert acc.cost_history() == []

    def test_record_and_totals(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        acc.record(StepCosts(time_step=2, surveillance_cost=3, response_cost=1, damage_cost=0))
        assert acc.total_surveillance == 13.0
        assert acc.total_response == 6.0
        assert acc.total_damage == 2.0
        assert acc.total_cost == 21.0
        assert acc.mean_cost_per_step() == 10.5

    def test_record_from_dict(self) -> None:
        acc = CostAccumulator()
        acc.record_from_dict(
            time_step=1,
            cost_dict={
                "surveillance_cost": 1.5,
                "response_cost": 2.5,
                "damage_cost": 3.5,
            },
        )
        assert len(acc.history) == 1
        assert acc.history[0].surveillance_cost == 1.5
        assert acc.history[0].response_cost == 2.5
        assert acc.history[0].damage_cost == 3.5

    def test_record_from_dict_missing_keys(self) -> None:
        acc = CostAccumulator()
        acc.record_from_dict(time_step=1, cost_dict={})
        assert acc.history[0].surveillance_cost == 0.0
        assert acc.history[0].response_cost == 0.0
        assert acc.history[0].damage_cost == 0.0

    def test_step_costs_total(self) -> None:
        c = StepCosts(time_step=1, surveillance_cost=1, response_cost=2, damage_cost=3)
        assert c.total == 6.0

    def test_history_methods(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        acc.record(StepCosts(time_step=2, surveillance_cost=3, response_cost=1, damage_cost=7))
        assert acc.surveillance_history() == [10.0, 3.0]
        assert acc.response_history() == [5.0, 1.0]
        assert acc.damage_history() == [2.0, 7.0]
        assert acc.cost_history() == [17.0, 11.0]

    def test_summary(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        s = acc.summary()
        assert s["total_surveillance_cost"] == 10.0
        assert s["total_response_cost"] == 5.0
        assert s["total_damage_cost"] == 2.0
        assert s["total_cost"] == 17.0
        assert s["mean_cost_per_step"] == 17.0
        assert s["steps_recorded"] == 1.0
