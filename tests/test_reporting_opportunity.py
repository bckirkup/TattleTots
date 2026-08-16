"""Tests for the reporting-opportunity measurement script."""

from __future__ import annotations

import pytest

from script_loading import load_script

script = load_script("measure_reporting_opportunity")


def _life(
    agent_id: str,
    adult_steps: int,
    escalations: int,
    attention_drift: float,
    death_currency: str = "attention",
) -> object:
    record = script.LifeRecord(agent_id=agent_id)
    record.adult_steps = adult_steps
    record.juvenile_steps = 5
    record.escalations = escalations
    record.reports = escalations
    record.steps_with_grounded_yield = adult_steps
    record.steps_above_threshold = escalations
    record.anomaly_gaps = [-0.5] * adult_steps
    record.attention_deltas = [attention_drift] * adult_steps
    record.information_deltas = [0.5] * adult_steps
    record.died = death_currency != ""
    record.death_currency = death_currency
    return record


def test_lifespan_metrics_attribute_deaths_to_the_exhausted_currency() -> None:
    """Death attribution follows which reserve was non-positive."""
    records = [
        _life("a", adult_steps=4, escalations=0, attention_drift=-0.1),
        _life("b", adult_steps=8, escalations=1, attention_drift=-0.1),
        _life("c", adult_steps=6, escalations=0, attention_drift=0.1, death_currency="information"),
    ]
    metrics = script.lifespan_metrics(records)
    assert metrics["mean_adult_steps"] == pytest.approx(6.0)
    assert metrics["deaths_by_attention"] == pytest.approx(2 / 3)
    assert metrics["deaths_by_information"] == pytest.approx(1 / 3)
    assert metrics["share_with_negative_attention_drift"] == pytest.approx(2 / 3)


def test_funnel_metrics_report_each_stage_as_a_share_of_adult_steps() -> None:
    """The funnel is measured per adult step, not per agent."""
    records = [
        _life("a", adult_steps=10, escalations=1, attention_drift=-0.1),
        _life("b", adult_steps=10, escalations=3, attention_drift=-0.1),
    ]
    funnel = script.funnel_metrics(records)
    assert funnel["adult_steps"] == pytest.approx(20.0)
    assert funnel["share_adult_steps_with_grounded_yield"] == pytest.approx(1.0)
    assert funnel["share_adult_steps_escalated"] == pytest.approx(0.2)
    assert funnel["median_anomaly_minus_threshold"] == pytest.approx(-0.5)


def test_silence_survival_coupling_separates_silent_from_reporting_adults() -> None:
    """Silent and reporting cohorts are summarised separately and correlated."""
    records = [
        _life(f"silent{i}", adult_steps=4, escalations=0, attention_drift=-0.2) for i in range(5)
    ]
    records += [
        _life(f"loud{i}", adult_steps=20, escalations=10, attention_drift=0.1) for i in range(5)
    ]
    coupling = script.silence_survival_coupling(records)
    assert coupling["silent_share_of_adults"] == pytest.approx(0.5)
    assert coupling["mean_adult_steps_silent"] == pytest.approx(4.0)
    assert coupling["mean_adult_steps_reporting"] == pytest.approx(20.0)
    assert coupling["corr_escalation_rate_adult_steps"] > 0.9
    assert coupling["corr_escalation_rate_attention_drift"] > 0.9


@pytest.mark.parametrize("rate", [0.05, 0.2, 0.8])
def test_counterfactual_requirements_scale_inversely_with_the_current_rate(rate: float) -> None:
    """A higher reporting rate needs fewer adult steps to reach the target."""
    ladder = [
        script.counterfactual_lifetime_reports(
            {"mean_adult_steps": 7.0}, {"reports_per_adult_step": r}, 7.2
        )["adult_steps_needed_at_current_rate"]
        for r in (0.05, 0.2, 0.8)
    ]
    assert ladder[0] > ladder[1] > ladder[2] > 0.0
    needed = script.counterfactual_lifetime_reports(
        {"mean_adult_steps": 7.0}, {"reports_per_adult_step": rate}, 7.2
    )
    assert needed["adult_steps_needed_at_current_rate"] == pytest.approx(7.2 / rate)
    assert needed["current_reports_per_lifetime"] == pytest.approx(rate * 7.0)


def test_empty_input_reports_zero_adults_instead_of_failing() -> None:
    """Every summary degrades to an adult count when nothing was observed."""
    assert script.lifespan_metrics([])["n_adults"] == pytest.approx(0.0)
    assert script.funnel_metrics([])["n_adults"] == pytest.approx(0.0)
    assert script.silence_survival_coupling([])["n_adults"] == pytest.approx(0.0)
