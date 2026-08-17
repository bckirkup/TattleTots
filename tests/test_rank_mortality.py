"""Tests for the rank-coupled-mortality measurement script."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import arm_sweep_fixtures as fixtures
from script_loading import load_script

script = load_script("measure_rank_mortality")


def _args(**overrides: Any) -> argparse.Namespace:
    lever: dict[str, Any] = {
        "threshold_range": (0.05, 0.3),
        "correctness_weight": 1.0,
        "recruitment_share": 1.0,
        "budget_scales": [0.5, 0.25],
    }
    lever.update(overrides)
    return fixtures.sweep_args(**lever)


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25, 0.1])
def test_arm_varies_only_the_attention_budget_scale(scale: float) -> None:
    """Every arm carries the same earlier levers; only the mortality pressure moves."""
    config = script.mortality_config(scale, _args())

    assert config["attention_budget_scale"] == pytest.approx(scale)
    assert config["reproduction_recruitment_share"] == pytest.approx(1.0)
    assert config["reproduction_correctness_weight"] == pytest.approx(1.0)
    assert config["reproduction_merit_ordering"] is True
    assert config["escalation_calibration_in_score_units"] is True
    assert config["correct_report_attention_value"] == pytest.approx(8.0)
    assert config["false_alarm_break_even_precision"] == pytest.approx(0.2)
    assert config["gene_pool"] == {"escalation_threshold_range": [0.05, 0.3]}


def test_arms_differ_from_each_other_only_in_the_budget_scale() -> None:
    args = _args()
    control = script.mortality_config(1.0, args)
    harsh = script.mortality_config(0.25, args)
    key = "attention_budget_scale"

    assert {name: value for name, value in control.items() if name != key} == {
        name: value for name, value in harsh.items() if name != key
    }


def test_pricing_lever_can_be_switched_off_without_leaking_a_key() -> None:
    config = script.mortality_config(0.5, _args(break_even_precision=None))
    assert "false_alarm_break_even_precision" not in config


def test_earlier_recruitment_lever_is_carried_through() -> None:
    config = script.mortality_config(0.5, _args(recruitment_share=0.25))
    assert config["reproduction_recruitment_share"] == pytest.approx(0.25)


def test_arm_configs_always_include_the_unscaled_control() -> None:
    labels = [label for label, _ in script.arm_configs(_args(budget_scales=[0.5]))]
    assert labels == ["budget_scale_1", "budget_scale_0.5"]


def test_arm_configs_are_ordered_from_mildest_to_harshest() -> None:
    labels = [label for label, _ in script.arm_configs(_args(budget_scales=[0.25, 1.0, 0.5, 0.25]))]
    assert labels == ["budget_scale_1", "budget_scale_0.5", "budget_scale_0.25"]


def _results(**overrides: Any) -> dict[str, Any]:
    arms = overrides.pop(
        "arms",
        dict(
            [
                fixtures.arm(
                    "budget_scale_1",
                    mean_rank_persistence=0.57,
                    mean_corr_rank_adult_steps=0.32,
                    mean_corr_parent_child_offspring=0.06,
                ),
                fixtures.arm(
                    "budget_scale_0.25",
                    mean_rank_persistence=0.48,
                    mean_corr_rank_adult_steps=0.40,
                    mean_corr_parent_child_offspring=0.29,
                ),
            ]
        ),
    )
    return fixtures.sweep_results(
        arms,
        correctness_weight=1.0,
        recruitment_share=1.0,
        budget_scales=[1.0, 0.25],
        **overrides,
    )


def test_markdown_report_renders_one_column_per_scale_with_the_rank_coupling() -> None:
    report = script.markdown_report(_results())

    assert "`budget_scale_1`" in report
    assert "`budget_scale_0.25`" in report
    assert "+0.480" in report
    assert "+0.400" in report
    assert "+0.290" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    results = _results(arms={"budget_scale_0.25": fixtures.empty_arm()})
    assert "n/a" in script.markdown_report(results)
