"""Tests for the escalation-threshold calibration measurement script."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import arm_sweep_fixtures as fixtures
from script_loading import load_script

script = load_script("measure_threshold_calibration")


def _args(**overrides: Any) -> argparse.Namespace:
    lever: dict[str, Any] = {"threshold_ranges": [(0.1, 0.5)]}
    lever.update(overrides)
    return fixtures.sweep_args(**lever)


def test_control_arm_differs_from_the_calibrated_arm_in_one_key() -> None:
    """The control and the score-unit arm share every other initial parameter."""
    args = _args()
    control = script.calibration_config(score_units=False, threshold_range=None, args=args)
    calibrated = script.calibration_config(score_units=True, threshold_range=None, args=args)

    assert control["escalation_calibration_in_score_units"] is False
    assert calibrated["escalation_calibration_in_score_units"] is True
    assert {
        key: value
        for key, value in control.items()
        if key != "escalation_calibration_in_score_units"
    } == {
        key: value
        for key, value in calibrated.items()
        if key != "escalation_calibration_in_score_units"
    }


@pytest.mark.parametrize("precision", [None, 0.1, 0.3])
def test_pricing_lever_is_carried_through_unchanged(precision: float | None) -> None:
    """The previous lever is held fixed, including when it is switched off."""
    config = script.calibration_config(
        score_units=True, threshold_range=None, args=_args(break_even_precision=precision)
    )
    if precision is None:
        assert "false_alarm_break_even_precision" not in config
    else:
        assert config["false_alarm_break_even_precision"] == pytest.approx(precision)


@pytest.mark.parametrize("bounds", [(0.1, 0.5), (0.05, 0.3), (0.2, 0.9)])
def test_threshold_range_arms_set_the_initial_trait_distribution(
    bounds: tuple[float, float],
) -> None:
    """A starting range travels as a gene-pool setting, not a simulation field."""
    config = script.calibration_config(score_units=True, threshold_range=bounds, args=_args())
    assert config["gene_pool"] == {"escalation_threshold_range": list(bounds)}


def test_arm_configs_cover_the_control_and_every_requested_range() -> None:
    configs = script.arm_configs(_args(threshold_ranges=[(0.1, 0.5), (0.05, 0.3)]))
    labels = [label for label, _ in configs]

    assert labels[0] == "raw_units_control"
    assert labels[1] == "score_units"
    assert labels[2:] == ["score_units_start_0.1_0.5", "score_units_start_0.05_0.3"]
    assert len({label for label in labels}) == len(labels)


def _results(**overrides: Any) -> dict[str, Any]:
    arms = overrides.pop(
        "arms",
        dict(
            [
                fixtures.arm("raw_units_control", mean_mean_reports_per_adult=0.45),
                fixtures.arm("score_units", mean_mean_reports_per_adult=3.5),
            ]
        ),
    )
    return fixtures.sweep_results(arms, **overrides)


def test_markdown_report_renders_one_column_per_arm() -> None:
    report = script.markdown_report(_results())

    assert "`raw_units_control`" in report
    assert "`score_units`" in report
    assert "0.45" in report
    assert "3.50" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    results = _results(arms={"score_units": fixtures.empty_arm()})
    assert "n/a" in script.markdown_report(results)
