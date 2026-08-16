"""Tests for the false-alarm pricing measurement script."""

from __future__ import annotations

import pytest

from script_loading import load_script

script = load_script("measure_false_alarm_pricing")


def testpricing_config_only_adds_the_target_when_one_is_given() -> None:
    """The flat-penalty arm differs from a priced arm in exactly one key."""
    flat = script.pricing_config(None, 8.0)
    priced = script.pricing_config(0.2, 8.0)

    assert "false_alarm_break_even_precision" not in flat
    assert priced["false_alarm_break_even_precision"] == pytest.approx(0.2)
    assert {key: priced[key] for key in flat} == flat


@pytest.mark.parametrize("value", [1.0, 4.0, 16.0])
def testpricing_config_carries_the_correct_report_value_through(value: float) -> None:
    """Correctness has to be worth something for a target to be priceable."""
    config = script.pricing_config(0.2, value)
    assert config["correct_report_attention_value"] == pytest.approx(value)
    assert config["reproduction_merit_ordering"] is True


def _arm(label: str, target: float | None, **summary: float) -> tuple[str, dict[str, object]]:
    return label, {"break_even_target": target, "summary": {"n_runs": 3, **summary}, "runs": []}


def test_markdown_report_renders_one_column_per_pricing_arm() -> None:
    """Every arm appears once, with its realized break-even precision."""
    results = {
        "adapter": "tattletots.scenarios.sparse_sensor:SparseSensorScenario",
        "arm": "ordinary",
        "steps": 200,
        "seeds": [42, 43],
        "grounded_input_fraction": 0.67,
        "max_population": 60,
        "correct_report_attention_value": 8.0,
        "arms": dict(
            [
                _arm("flat_penalty", None, mean_realized_break_even_precision=0.8),
                _arm("break_even_0.2", 0.2, mean_realized_break_even_precision=0.2),
            ]
        ),
    }
    report = script.markdown_report(results)

    assert "`flat_penalty`" in report
    assert "`break_even_0.2`" in report
    assert "80.00%" in report
    assert "20.00%" in report


def test_markdown_report_marks_empty_arms_instead_of_printing_zeros() -> None:
    """An arm with no surviving adults is n/a, not a fabricated 0.00%."""
    results = {
        "adapter": "adapter",
        "arm": "ordinary",
        "steps": 10,
        "seeds": [42],
        "grounded_input_fraction": 0.67,
        "max_population": 60,
        "correct_report_attention_value": 8.0,
        "arms": {"flat_penalty": {"break_even_target": None, "summary": {"n_runs": 0}, "runs": []}},
    }
    assert "n/a" in script.markdown_report(results)
