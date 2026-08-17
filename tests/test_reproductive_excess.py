"""Tests for the reproductive-excess measurement script."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import arm_sweep_fixtures as fixtures
from script_loading import load_script

script = load_script("measure_reproductive_excess")


def _args(**overrides: Any) -> argparse.Namespace:
    lever: dict[str, Any] = {
        "threshold_range": (0.05, 0.3),
        "correctness_weight": 1.0,
        "shares": [0.5, 0.25],
    }
    lever.update(overrides)
    return fixtures.sweep_args(**lever)


@pytest.mark.parametrize("share", [1.0, 0.5, 0.25, 0.1])
def test_excess_arm_varies_only_the_recruitment_share(share: float) -> None:
    """Every arm carries the same earlier levers; only the recruitment limit moves."""
    config = script.excess_config(share, _args())

    assert config["reproduction_recruitment_share"] == pytest.approx(share)
    assert config["reproduction_correctness_weight"] == pytest.approx(1.0)
    assert config["reproduction_merit_ordering"] is True
    assert config["escalation_calibration_in_score_units"] is True
    assert config["correct_report_attention_value"] == pytest.approx(8.0)
    assert config["false_alarm_break_even_precision"] == pytest.approx(0.2)
    assert config["gene_pool"] == {"escalation_threshold_range": [0.05, 0.3]}


def test_excess_arms_differ_from_each_other_only_in_the_share() -> None:
    args = _args()
    control = script.excess_config(1.0, args)
    scarce = script.excess_config(0.25, args)
    key = "reproduction_recruitment_share"

    assert {name: value for name, value in control.items() if name != key} == {
        name: value for name, value in scarce.items() if name != key
    }


def test_pricing_lever_can_be_switched_off_without_leaking_a_key() -> None:
    config = script.excess_config(0.5, _args(break_even_precision=None))
    assert "false_alarm_break_even_precision" not in config


def test_arm_configs_always_include_the_unlimited_recruitment_control() -> None:
    labels = [label for label, _ in script.arm_configs(_args(shares=[0.5]))]
    assert labels == ["recruitment_share_1", "recruitment_share_0.5"]


def test_arm_configs_are_ordered_from_least_to_most_scarce() -> None:
    labels = [label for label, _ in script.arm_configs(_args(shares=[0.25, 1.0, 0.5, 0.25]))]
    assert labels == [
        "recruitment_share_1",
        "recruitment_share_0.5",
        "recruitment_share_0.25",
    ]


def _results(**overrides: Any) -> dict[str, Any]:
    arms = overrides.pop(
        "arms",
        dict(
            [
                fixtures.arm(
                    "recruitment_share_1",
                    mean_reproductive_excess=1.02,
                    mean_corr_parent_child_offspring=0.07,
                ),
                fixtures.arm(
                    "recruitment_share_0.25",
                    mean_reproductive_excess=4.10,
                    mean_corr_parent_child_offspring=0.31,
                ),
            ]
        ),
    )
    return fixtures.sweep_results(arms, correctness_weight=1.0, shares=[1.0, 0.25], **overrides)


def test_markdown_report_renders_one_column_per_share_with_the_excess() -> None:
    report = script.markdown_report(_results())

    assert "`recruitment_share_1`" in report
    assert "`recruitment_share_0.25`" in report
    assert "4.10" in report
    assert "+0.310" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    results = _results(arms={"recruitment_share_0.25": fixtures.empty_arm()})
    assert "n/a" in script.markdown_report(results)
