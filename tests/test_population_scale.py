"""Tests for the population-scale measurement script."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import arm_sweep_fixtures as fixtures
from script_loading import load_script

script = load_script("measure_population_scale")


def _args(**overrides: Any) -> argparse.Namespace:
    lever: dict[str, Any] = {
        "threshold_range": (0.05, 0.3),
        "caps": [60, 125, 250],
        "reference_cap": 60,
    }
    lever.update(overrides)
    return fixtures.sweep_args(**lever)


@pytest.mark.parametrize("cap", [20, 60, 125, 250, 600])
def test_founding_population_scales_with_the_cap_and_stays_viable(cap: int) -> None:
    """Founding diversity is a fixed share of the cap, and never below a breeding pair."""
    founders = script.founding_population(cap)

    assert founders >= 2
    assert founders <= cap


def test_founding_population_is_monotone_in_the_cap() -> None:
    caps = [20, 60, 125, 250, 600]
    founders = [script.founding_population(cap) for cap in caps]

    assert founders == sorted(founders)
    assert founders[0] < founders[-1]


@pytest.mark.parametrize("cap", [60, 125, 250])
def test_population_arm_varies_only_the_population_terms(cap: int) -> None:
    """Every arm carries the same earlier levers; only the population terms move."""
    config = script.population_config(cap, _args())

    assert config["max_population"] == cap
    assert config["initial_population"] == script.founding_population(cap)
    assert config["reproduction_merit_ordering"] is True
    assert config["escalation_calibration_in_score_units"] is True
    assert config["correct_report_attention_value"] == pytest.approx(8.0)
    assert config["false_alarm_break_even_precision"] == pytest.approx(0.2)
    assert config["gene_pool"] == {"escalation_threshold_range": [0.05, 0.3]}


def test_pricing_lever_can_be_switched_off_without_leaking_a_key() -> None:
    config = script.population_config(60, _args(break_even_precision=None))
    assert "false_alarm_break_even_precision" not in config


def test_population_arms_differ_from_each_other_only_in_the_population_terms() -> None:
    args = _args()
    small = script.population_config(60, args)
    large = script.population_config(250, args)
    population_keys = {"max_population", "initial_population"}

    assert {key: value for key, value in small.items() if key not in population_keys} == {
        key: value for key, value in large.items() if key not in population_keys
    }


def test_arm_configs_are_labelled_by_cap_in_ascending_order() -> None:
    configs = script.arm_configs(_args(caps=[250, 60, 125]))
    labels = [label for label, _ in configs]

    assert labels[:3] == ["cap_60", "cap_125", "cap_250"]
    assert labels[3:] == ["cap_125_per_capita", "cap_250_per_capita"]
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize(("cap", "expected"), [(60, 1.0), (125, 125 / 60), (250, 250 / 60)])
def test_per_capita_arms_scale_attention_with_the_cap(cap: int, expected: float) -> None:
    """Holding per-capita solvency fixed means the budget grows in step with the cap."""
    config = script.population_config(cap, _args(), per_capita_attention=True)
    assert config["attention_budget_scale"] == pytest.approx(expected)


def test_fixed_budget_arms_do_not_scale_attention() -> None:
    config = script.population_config(250, _args())
    assert "attention_budget_scale" not in config


def test_reference_cap_gets_no_redundant_per_capita_arm() -> None:
    """At the reference cap the two arms would be identical, so only one is run."""
    labels = [label for label, _ in script.arm_configs(_args(caps=[60, 250], reference_cap=60))]
    assert "cap_60_per_capita" not in labels


def _results(**overrides: Any) -> dict[str, Any]:
    arms = overrides.pop(
        "arms",
        dict(
            [
                fixtures.arm("cap_60", mean_final_population=48.0),
                fixtures.arm("cap_250", mean_final_population=191.0),
            ]
        ),
    )
    return fixtures.sweep_results(arms, caps=[60, 250], reference_cap=60, **overrides)


def test_markdown_report_renders_one_column_per_cap() -> None:
    report = script.markdown_report(_results())

    assert "`cap_60`" in report
    assert "`cap_250`" in report
    assert "48.0" in report
    assert "191.0" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    results = _results(arms={"cap_250": fixtures.empty_arm()})
    assert "n/a" in script.markdown_report(results)
