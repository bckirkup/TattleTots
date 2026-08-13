"""Unit tests for telemetry/recorder.py and telemetry/cost_accounting.py."""

from __future__ import annotations

import pytest

from tattletots.models.location import EventLocation
from tattletots.output_schema import EcologyMetrics, RunSummary, SimulationOutput, TimeSeries
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
    **kwargs: float | int | bool | tuple[EventLocation, ...],
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
        assert rec.max_trophic_depth == pytest.approx(0.0)
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
        assert rec.max_trophic_depth == pytest.approx(3.5)

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
            "event_prevalence",
            "chance_precision",
            "static_prior_precision",
            "location_support_size",
            "grounded_yield_share",
            "effective_grounded_yield_share",
            "attention_solvent_fraction",
            "mean_attention_carrying_capacity",
            "initiation_is_degenerate",
            "initiation_degeneracy_reasons",
            "max_trophic_depth",
            "reached_equilibrium",
            "total_responses_dispatched",
            "total_responses_judged_necessary",
            "total_responses_judged_unnecessary",
            "responder_necessity_rate",
            "unnecessary_dispatch_rate",
            "designed_population_share",
            "designed_precision",
            "ordinary_precision",
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
        assert h["info_yield"] == [pytest.approx(5.0)]
        assert h["attn_income"] == [pytest.approx(3.0)]
        assert h["compute_cost"] == [pytest.approx(1.0)]
        assert h["maintenance_cost"] == [pytest.approx(0.5)]

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
        assert d["juveniles"] == [pytest.approx(3.0)]
        assert d["adults"] == [pytest.approx(7.0)]
        assert d["mean_generation"] == [pytest.approx(2.5)]
        assert d["compression_types"] == [pytest.approx(3.0)]

    def test_ecology_time_series(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                time_step=1,
                population=10,
                births=2,
                deaths=1,
                reports_issued=5,
                correct_reports=4,
                false_alarms=1,
                missed_events=2,
                mean_info_energy=1.5,
                mean_attn_energy=0.8,
                max_trophic_level=2.0,
                n_compression_types=3,
            )
        )
        rec.record_step(
            _make_record(
                time_step=2,
                population=11,
                births=1,
                deaths=0,
                reports_issued=2,
                correct_reports=2,
                false_alarms=0,
                missed_events=0,
                mean_info_energy=1.6,
                mean_attn_energy=0.9,
                max_trophic_level=2.5,
                n_compression_types=4,
            )
        )
        ts = rec.ecology_time_series()
        assert ts["population"] == [10, 11]
        assert ts["births"] == [2, 1]
        assert ts["deaths"] == [1, 0]
        assert ts["reports_issued"] == [5, 2]
        assert ts["correct_reports"] == [4, 2]
        assert ts["false_alarms"] == [1, 0]
        assert ts["missed_events"] == [2, 0]
        assert ts["mean_info_energy"] == [pytest.approx(1.5), pytest.approx(1.6)]
        assert ts["mean_attn_energy"] == [pytest.approx(0.8), pytest.approx(0.9)]
        assert ts["n_compression_types"] == [3, 4]
        assert ts["max_trophic_level"] == [pytest.approx(2.0), pytest.approx(2.5)]
        assert ts["responses_dispatched"] == [0, 0]
        assert ts["responses_judged_necessary"] == [0, 0]
        assert ts["responses_judged_unnecessary"] == [0, 0]
        assert ts["n_attention_solvent_agents"] == [0, 0]
        assert ts["n_attention_eligible_agents"] == [0, 0]
        assert ts["attention_carrying_capacity"] == [pytest.approx(0.0), pytest.approx(0.0)]
        assert ts["grounded_info_yield"] == [pytest.approx(0.0), pytest.approx(0.0)]
        assert ts["ungrounded_info_yield"] == [pytest.approx(0.0), pytest.approx(0.0)]
        assert ts["grounded_yield_share"] == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_time_series_from_telemetry(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(_make_record(time_step=1, population=10, reports_issued=3))
        rec.record_step(_make_record(time_step=2, population=9, reports_issued=1))
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=1, response_cost=2, damage_cost=3))
        acc.record(StepCosts(time_step=2, surveillance_cost=4, response_cost=5, damage_cost=6))

        ts = TimeSeries.from_telemetry(rec, acc.cost_history())
        assert ts.population == [10, 9]
        assert ts.reports_issued == [3, 1]
        assert ts.cost_per_step == [pytest.approx(6.0), pytest.approx(15.0)]

    def test_reporter_groups_round_trip_through_simulation_output(self, tmp_path) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(time_step=1, reports_issued=3, correct_reports=2),
            reporter_groups={
                "designed_population_share": 0.5,
                "designed_reports": 2,
                "ordinary_reports": 1,
                "designed_correct_reports": 1,
                "ordinary_correct_reports": 1,
            },
        )
        rec.record_step(
            _make_record(time_step=2, reports_issued=2, correct_reports=1),
            reporter_groups={
                "designed_population_share": 0.75,
                "designed_reports": 1,
                "ordinary_reports": 1,
                "designed_correct_reports": 0,
                "ordinary_correct_reports": 1,
            },
        )
        summary = rec.summary()
        output = SimulationOutput(
            run_summary=RunSummary(domain="test", steps_completed=2),
            ecology_metrics=EcologyMetrics(
                designed_population_share=summary["designed_population_share"],
                designed_precision=summary["designed_precision"],
                ordinary_precision=summary["ordinary_precision"],
            ),
            time_series=TimeSeries.from_telemetry(rec, [0.0, 0.0]),
        )

        path = tmp_path / "simulation-output.json"
        output.write_json(path)
        loaded = SimulationOutput.read_json(path)

        assert loaded.ecology_metrics.designed_population_share == pytest.approx(0.625)
        assert loaded.ecology_metrics.designed_precision == pytest.approx(1 / 3)
        assert loaded.ecology_metrics.ordinary_precision == pytest.approx(1.0)
        assert loaded.time_series.designed_population_share == [0.5, 0.75]
        assert loaded.time_series.designed_reports == [2, 1]
        assert loaded.time_series.ordinary_reports == [1, 1]
        assert loaded.time_series.designed_correct_reports == [1, 0]
        assert loaded.time_series.ordinary_correct_reports == [1, 1]
        assert len(rec.history) == len(rec.reporter_group_history) == 2

    def test_initiation_degeneracy_names_each_reason(self) -> None:
        cases = (
            (
                "grounded_yield_share_below_minimum",
                _make_record(
                    reports_issued=1,
                    correct_reports=1,
                    population=1,
                    n_attention_solvent_agents=1,
                    ungrounded_info_yield=1.0,
                ),
            ),
            (
                "attention_insolvency_with_capacity_overshoot",
                _make_record(
                    births=1,
                    reports_issued=1,
                    correct_reports=1,
                    population=1,
                    n_attention_solvent_agents=0,
                    n_attention_eligible_agents=1,
                    attention_carrying_capacity=0.5,
                    grounded_info_yield=1.0,
                ),
            ),
        )
        for reason, record in cases:
            rec = TelemetryRecorder()
            rec.record_step(record)
            degenerate, reasons = rec.initiation_degeneracy()
            assert degenerate
            assert reason in reasons

    def test_initiation_threshold_sensitivity(self) -> None:
        records = [
            _make_record(
                time_step=step,
                active_location_count=1,
                ground_truth_locations=(location,),
                verified_report_locations=(location,),
                reports_issued=1,
                correct_reports=1,
                population=1,
                n_attention_solvent_agents=1,
                grounded_info_yield=1.0,
                ungrounded_info_yield=1.0,
            )
            for step, location in enumerate(((0, 0), (1, 1)))
        ]
        permissive = TelemetryRecorder()
        for record in records:
            permissive.record_step(record)
        strict = TelemetryRecorder()
        strict.configure_initiation_thresholds(
            min_grounded_yield_share=0.75,
            attention_insolvency_steps_fraction=0.8,
        )
        for record in records:
            strict.record_step(record)
        assert not permissive.initiation_degeneracy()[0]
        assert strict.initiation_degeneracy()[0]
        assert strict.summary()["grounded_yield_share"] == pytest.approx(0.5)

    def test_static_prior_precision_reason_uses_report_timing(self) -> None:
        rec = TelemetryRecorder()
        for step, location in enumerate(((0, 0), (1, 1))):
            rec.record_step(
                _make_record(
                    time_step=step,
                    active_location_count=1,
                    ground_truth_locations=(location,),
                    reports_issued=1,
                    correct_reports=0,
                    population=1,
                    n_attention_solvent_agents=1,
                    n_attention_eligible_agents=1,
                    grounded_info_yield=1.0,
                )
            )

        degenerate, reasons = rec.initiation_degeneracy()

        assert degenerate
        assert "precision_not_above_static_prior" in reasons

    def test_attention_degeneracy_threshold_and_capacity_sensitivity(self) -> None:
        records = [
            _make_record(
                time_step=step,
                active_location_count=1,
                ground_truth_locations=(location,),
                verified_report_locations=(location,),
                population=2,
                births=1,
                reports_issued=1,
                correct_reports=1,
                n_attention_solvent_agents=1,
                n_attention_eligible_agents=2,
                attention_carrying_capacity=1.0,
                grounded_info_yield=1.0,
            )
            for step, location in enumerate(((0, 0), (1, 1)))
        ]
        baseline = TelemetryRecorder()
        for record in records:
            baseline.record_step(record)
        stricter_solvent = TelemetryRecorder(initiation_min_solvent_fraction=0.75)
        for record in records:
            stricter_solvent.record_step(record)
        stricter_capacity = TelemetryRecorder(initiation_population_capacity_overshoot_factor=3.0)
        for record in records:
            stricter_capacity.record_step(record)
        assert not baseline.initiation_degeneracy()[0]
        assert stricter_solvent.initiation_degeneracy()[0]
        assert not stricter_capacity.initiation_degeneracy()[0]

    def test_seeded_telemetry_golden_values(self) -> None:
        record = _make_record(
            population=4,
            reports_issued=4,
            correct_reports=2,
            ground_truth_active=True,
            active_location_count=1,
            ground_truth_locations=((0, 0), (1, 1)),
            verified_report_locations=((0, 0),),
            n_attention_solvent_agents=2,
            n_attention_eligible_agents=4,
            attention_carrying_capacity=8.0,
            grounded_info_yield=3.0,
            ungrounded_info_yield=1.0,
        )
        rec = TelemetryRecorder()
        rec.record_step(record)
        summary = rec.summary()
        assert summary["precision"] == pytest.approx(0.5)
        assert summary["event_prevalence"] == pytest.approx(1.0)
        assert summary["chance_precision"] == pytest.approx(0.5)
        assert summary["location_support_size"] == 2
        assert summary["grounded_yield_share"] == pytest.approx(0.75)
        assert summary["attention_solvent_fraction"] == pytest.approx(0.5)
        assert summary["mean_attention_carrying_capacity"] == pytest.approx(8.0)

    def test_agents_without_attention_delta_are_excluded(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                population=2,
                births=1,
                reports_issued=1,
                correct_reports=1,
                n_attention_solvent_agents=1,
                n_attention_eligible_agents=1,
                attention_carrying_capacity=10.0,
                grounded_info_yield=1.0,
            )
        )
        assert rec.summary()["attention_solvent_fraction"] == pytest.approx(1.0)
        assert (
            "attention_insolvency_with_capacity_overshoot"
            not in (rec.summary()["initiation_degeneracy_reasons"])
        )

    def test_chance_baseline_does_not_flag_good_prevalence_one_reporting(self) -> None:
        rec = TelemetryRecorder()
        for step, location in enumerate(((0, 0), (1, 1))):
            rec.record_step(
                _make_record(
                    time_step=step,
                    population=2,
                    reports_issued=1,
                    correct_reports=1,
                    active_location_count=1,
                    ground_truth_active=True,
                    ground_truth_locations=(location,),
                    verified_report_locations=(location,),
                    n_attention_solvent_agents=2,
                    n_attention_eligible_agents=2,
                    grounded_info_yield=1.0,
                    attention_carrying_capacity=10.0,
                )
            )

        degenerate, reasons = rec.initiation_degeneracy()

        assert not degenerate
        assert "precision_not_above_static_prior" not in reasons
        assert rec.summary()["event_prevalence"] == pytest.approx(1.0)
        assert rec.summary()["chance_precision"] == pytest.approx(0.5)
        assert rec.summary()["static_prior_precision"] == pytest.approx(0.5)
        assert rec.summary()["location_support_size"] == 2

    def test_static_prior_baseline_replaces_uniform_precision_null(self) -> None:
        rec = TelemetryRecorder()
        for step in range(5):
            location = (0, 0) if step < 4 else (1, 1)
            rec.record_step(
                _make_record(
                    time_step=step,
                    population=2,
                    reports_issued=5,
                    correct_reports=2,
                    active_location_count=1,
                    ground_truth_active=True,
                    ground_truth_locations=(location,),
                    n_attention_solvent_agents=2,
                    n_attention_eligible_agents=2,
                    grounded_info_yield=1.0,
                    attention_carrying_capacity=10.0,
                )
            )

        degenerate, reasons = rec.initiation_degeneracy()

        assert degenerate
        assert "precision_not_above_static_prior" in reasons
        assert rec.summary()["chance_precision"] == pytest.approx(0.5)
        assert rec.summary()["static_prior_precision"] == pytest.approx(0.8)

    def test_zero_event_window_has_distinct_reason(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                reports_issued=2,
                correct_reports=0,
                verified_report_locations=((0, 0), (1, 1)),
                grounded_info_yield=1.0,
            )
        )

        degenerate, reasons = rec.initiation_degeneracy()

        assert degenerate
        assert "no_ground_truth_events" in reasons
        assert "precision_not_above_static_prior" not in reasons
        assert rec.summary()["event_prevalence"] == pytest.approx(0.0)

    def test_trivial_location_support_has_distinct_reason(self) -> None:
        rec = TelemetryRecorder()
        rec.record_step(
            _make_record(
                active_location_count=1,
                ground_truth_active=True,
                ground_truth_locations=((0, 0),),
                verified_report_locations=((0, 0),),
                reports_issued=1,
                correct_reports=1,
                grounded_info_yield=1.0,
            )
        )

        degenerate, reasons = rec.initiation_degeneracy()

        assert degenerate
        assert "insufficient_location_support" in reasons
        assert "localization_vacuous" in reasons
        assert "precision_not_above_static_prior" not in reasons


class TestCostAccumulator:
    def test_empty_accumulator(self) -> None:
        acc = CostAccumulator()
        assert acc.total_surveillance == pytest.approx(0.0)
        assert acc.total_response == pytest.approx(0.0)
        assert acc.total_damage == pytest.approx(0.0)
        assert acc.total_cost == pytest.approx(0.0)
        assert acc.mean_cost_per_step() == pytest.approx(0.0)
        assert acc.cost_history() == []

    def test_record_and_totals(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        acc.record(StepCosts(time_step=2, surveillance_cost=3, response_cost=1, damage_cost=0))
        assert acc.total_surveillance == pytest.approx(13.0)
        assert acc.total_response == pytest.approx(6.0)
        assert acc.total_damage == pytest.approx(2.0)
        assert acc.total_cost == pytest.approx(21.0)
        assert acc.mean_cost_per_step() == pytest.approx(10.5)

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
        assert acc.history[0].surveillance_cost == pytest.approx(1.5)
        assert acc.history[0].response_cost == pytest.approx(2.5)
        assert acc.history[0].damage_cost == pytest.approx(3.5)

    def test_record_from_dict_missing_keys(self) -> None:
        acc = CostAccumulator()
        acc.record_from_dict(time_step=1, cost_dict={})
        assert acc.history[0].surveillance_cost == pytest.approx(0.0)
        assert acc.history[0].response_cost == pytest.approx(0.0)
        assert acc.history[0].damage_cost == pytest.approx(0.0)

    def test_step_costs_total(self) -> None:
        c = StepCosts(time_step=1, surveillance_cost=1, response_cost=2, damage_cost=3)
        assert c.total == pytest.approx(6.0)

    def test_history_methods(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        acc.record(StepCosts(time_step=2, surveillance_cost=3, response_cost=1, damage_cost=7))
        assert acc.surveillance_history() == [pytest.approx(10.0), pytest.approx(3.0)]
        assert acc.response_history() == [pytest.approx(5.0), pytest.approx(1.0)]
        assert acc.damage_history() == [pytest.approx(2.0), pytest.approx(7.0)]
        assert acc.cost_history() == [pytest.approx(17.0), pytest.approx(11.0)]

    def test_summary(self) -> None:
        acc = CostAccumulator()
        acc.record(StepCosts(time_step=1, surveillance_cost=10, response_cost=5, damage_cost=2))
        s = acc.summary()
        assert s["total_surveillance_cost"] == pytest.approx(10.0)
        assert s["total_response_cost"] == pytest.approx(5.0)
        assert s["total_damage_cost"] == pytest.approx(2.0)
        assert s["total_cost"] == pytest.approx(17.0)
        assert s["mean_cost_per_step"] == pytest.approx(17.0)
        assert s["steps_recorded"] == pytest.approx(1.0)
