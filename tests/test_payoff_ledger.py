"""Tests for the per-agent payoff ledger."""

from __future__ import annotations

import math

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.scenarios.sparse_sensor import SparseSensorScenario
from tattletots.telemetry.payoff_ledger import PayoffLedger

_STEPS = 40
_CORRELATION_KEYS = (
    "corr_correct_reports_trust",
    "corr_precision_trust",
    "corr_trust_attention_income",
    "corr_correct_reports_attention_income",
    "corr_correct_reports_information_income",
    "corr_attention_income_offspring",
    "corr_information_income_offspring",
    "corr_correct_reports_offspring",
    "corr_reports_issued_attention_income",
    "corr_reports_issued_offspring",
)


def _run(
    attention_budget_scale: float = 1.0,
    seed: int = 42,
    correct_report_attention_value: float = 0.0,
) -> tuple[World, PayoffLedger]:
    adapter = SparseSensorScenario(seed=seed, total_steps=_STEPS)
    config = SimulationConfig(
        initial_population=10,
        max_population=24,
        max_steps=_STEPS,
        seed=seed,
        grounded_input_fraction=0.67,
        correct_report_attention_value=correct_report_attention_value,
    )
    world = World(config=config)
    for stream in adapter.get_streams():
        world.add_stream(stream)
    for user in adapter.get_users():
        user.attention_budget *= attention_budget_scale
        world.add_user(user)
    world.seed_population()
    world.set_location_inference(adapter.infer_report_location)
    world.set_location_frame(adapter.get_location_frame())

    ledger = PayoffLedger()
    for step in range(_STEPS):
        adapter.step(step)
        world.set_event_state(adapter.get_active_locations(step))
        world.step()
        ledger.observe(world)
    ledger.finalize(world)
    return world, ledger


def test_ledger_metrics_are_bounded_and_finite() -> None:
    _, ledger = _run()
    summary = ledger.coupling_summary()

    assert summary["n_adults"] > 0
    for key in _CORRELATION_KEYS:
        assert -1.0 <= summary[key] <= 1.0
    for key in ("mean_subsidy_share_of_information_income", "trust_break_even_precision"):
        assert 0.0 <= summary[key] <= 1.0
    # Reserve share can exceed 1.0 when an agent's attention reserve goes negative.
    assert math.isfinite(summary["mean_information_share_of_reserves"])
    for share in summary["reproduction_gate"].values():
        assert 0.0 <= share <= 1.0


def test_ledger_conserves_offspring_and_report_counts() -> None:
    world, ledger = _run()
    records = {record.agent_id: record for record in ledger.records}

    assert set(records) == set(world.agents)
    expected_offspring = sum(len(agent.state.parent_ids) for agent in world.agents.values())
    assert sum(record.offspring for record in records.values()) <= expected_offspring
    for agent in world.agents.values():
        record = records[agent.id]
        assert record.correct_reports + record.false_alarms <= record.reports_issued
        assert record.steps_alive >= record.adult_steps


def test_attention_income_rises_with_user_attention_budget() -> None:
    incomes = []
    couplings = []
    for scale in (1.0, 5.0, 25.0):
        _, ledger = _run(attention_budget_scale=scale)
        summary = ledger.coupling_summary()
        incomes.append(summary["mean_attention_income_per_step"])
        couplings.append(summary["corr_trust_attention_income"])

    assert incomes[0] < incomes[1] < incomes[2]
    assert couplings[-1] >= couplings[0]


def test_information_dominates_reserves_at_default_attention_budget() -> None:
    """The correctness-blind currency supplies most of the reproduction reserve."""
    _, ledger = _run()
    summary = ledger.coupling_summary()

    assert summary["mean_information_income_per_step"] > summary["mean_attention_income_per_step"]
    assert summary["mean_information_share_of_reserves"] > 0.5


def test_correct_reporters_earn_more_attention_as_verified_value_rises() -> None:
    """The knob prices correctness into attention income; default pays nothing for it."""
    correct_income = []
    silent_income = []
    for value in (0.0, 4.0, 16.0):
        _, ledger = _run(correct_report_attention_value=value)
        summary = ledger.coupling_summary()
        correct_income.append(summary["correct_group_mean_attention_income"])
        silent_income.append(summary["never_correct_group_mean_attention_income"])

    assert correct_income[0] < correct_income[1] < correct_income[2]
    assert correct_income[0] < silent_income[0]
    assert correct_income[-1] > silent_income[-1]


def test_falsification_metrics_are_reported_and_bounded() -> None:
    """The two clauses are measurable on the same run that measures the links."""
    _, ledger = _run()
    summary = ledger.coupling_summary()

    assert summary["generations_observed"] >= 1
    assert summary["n_parent_child_pairs"] > 0
    assert -1.0 <= summary["corr_parent_child_offspring"] <= 1.0
    assert -1.0 <= summary["corr_parent_child_precision"] <= 1.0
    assert math.isfinite(summary["precision_generation_slope"])
    for rate in summary["precision_by_generation"].values():
        assert 0.0 <= rate <= 1.0


def test_false_alarm_cost_dwarfs_attention_income_at_default_config() -> None:
    _, ledger = _run()
    summary = ledger.coupling_summary()

    assert summary["false_alarm_penalty_in_attention_income_steps"] > 1.0
