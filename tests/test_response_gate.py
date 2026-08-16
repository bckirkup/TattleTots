"""Tests for the response-gate measurement script."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import arm_sweep_fixtures as fixtures
from script_loading import load_script

script = load_script("measure_response_gate")


def _args(**overrides: Any) -> argparse.Namespace:
    lever: dict[str, Any] = {"threshold_range": (0.05, 0.3), "weights": [0.25, 1.0]}
    lever.update(overrides)
    return fixtures.sweep_args(**lever)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_gate_arm_varies_only_the_correctness_weight(weight: float) -> None:
    """Every arm carries the same earlier levers; only the merit mix moves."""
    config = script.gate_config(weight, _args())

    assert config["reproduction_correctness_weight"] == pytest.approx(weight)
    assert config["reproduction_merit_ordering"] is True
    assert config["escalation_calibration_in_score_units"] is True
    assert config["correct_report_attention_value"] == pytest.approx(8.0)
    assert config["false_alarm_break_even_precision"] == pytest.approx(0.2)
    assert config["gene_pool"] == {"escalation_threshold_range": [0.05, 0.3]}


def test_gate_arms_differ_from_each_other_only_in_the_weight() -> None:
    args = _args()
    control = script.gate_config(0.0, args)
    weighted = script.gate_config(1.0, args)
    key = "reproduction_correctness_weight"

    assert {name: value for name, value in control.items() if name != key} == {
        name: value for name, value in weighted.items() if name != key
    }


def test_pricing_lever_can_be_switched_off_without_leaking_a_key() -> None:
    config = script.gate_config(0.5, _args(break_even_precision=None))
    assert "false_alarm_break_even_precision" not in config


def test_arm_configs_always_include_the_reserves_only_control() -> None:
    labels = [label for label, _ in script.arm_configs(_args(weights=[0.5]))]
    assert labels == ["correctness_weight_0", "correctness_weight_0.5"]


def test_arm_configs_are_ordered_and_deduplicated() -> None:
    labels = [label for label, _ in script.arm_configs(_args(weights=[1.0, 0.25, 0.25, 0.0]))]
    assert labels == [
        "correctness_weight_0",
        "correctness_weight_0.25",
        "correctness_weight_1",
    ]


def _results(**overrides: Any) -> dict[str, Any]:
    arms = overrides.pop(
        "arms",
        dict(
            [
                fixtures.arm("correctness_weight_0", mean_corr_parent_child_offspring=0.07),
                fixtures.arm("correctness_weight_1", mean_corr_parent_child_offspring=0.31),
            ]
        ),
    )
    return fixtures.sweep_results(arms, weights=[0.0, 1.0], **overrides)


def test_markdown_report_renders_one_column_per_weight() -> None:
    report = script.markdown_report(_results())

    assert "`correctness_weight_0`" in report
    assert "`correctness_weight_1`" in report
    assert "+0.070" in report
    assert "+0.310" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    results = _results(arms={"correctness_weight_1": fixtures.empty_arm()})
    assert "n/a" in script.markdown_report(results)
