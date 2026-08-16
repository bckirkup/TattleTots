#!/usr/bin/env python3
"""Measure what calibrating escalation thresholds in score units buys.

`docs/reporting-opportunity-measurement.md` found the escalation threshold sitting well
above the anomaly distribution it is compared against. One reason is mechanical: the
adaptive modes in `engine/escalation.py` calibrate on the *raw* anomaly window, whose
scale is set by the compression model, while `should_escalate` compares the *normalized*
0-1 score against the result, so the quantile and volatility traits do not control firing
rate in the units the decision uses.

This script sweeps `SimulationConfig.escalation_calibration_in_score_units` (and the
starting threshold range, which sets where the fixed-mode agents begin) at otherwise fixed
initial parameters, reporting reporting economics and both falsification clauses so this
lever's contribution stays separable from the pricing lever measured in
`docs/false-alarm-pricing-measurement.md`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/threshold-calibration.json"
_REPORT_ARTIFACT = "docs/threshold-calibration.md"

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def calibration_config(
    *,
    score_units: bool,
    threshold_range: tuple[float, float] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Config overrides for one calibration arm, holding the pricing lever fixed."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": args.correct_report_value,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": score_units,
    }
    if args.break_even_precision is not None:
        extra["false_alarm_break_even_precision"] = args.break_even_precision
    if threshold_range is not None:
        extra["gene_pool"] = {"escalation_threshold_range": list(threshold_range)}
    return extra


def arm_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """The calibration control, the score-unit arm, and lowered starting thresholds."""
    configs: list[tuple[str, dict[str, Any]]] = [
        (
            "raw_units_control",
            calibration_config(score_units=False, threshold_range=None, args=args),
        ),
        (
            "score_units",
            calibration_config(score_units=True, threshold_range=None, args=args),
        ),
    ]
    for low, high in args.threshold_ranges:
        configs.append(
            (
                f"score_units_start_{low:g}_{high:g}",
                calibration_config(score_units=True, threshold_range=(low, high), args=args),
            )
        )
    return configs


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every calibration arm at the same fixed initial parameters."""
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs(args))
    results["correct_report_attention_value"] = args.correct_report_value
    results["break_even_precision"] = args.break_even_precision
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the calibration sweep as one table of arms against reporting economics."""
    preamble = [
        (
            "- `correct_report_attention_value` (fixed across arms): "
            f"`{results['correct_report_attention_value']:g}`"
        ),
        f"- `false_alarm_break_even_precision` (fixed across arms): "
        f"`{results['break_even_precision']}`",
        "",
        "Every arm shares the same initial parameters; the arms differ in whether adaptive "
        "escalation thresholds are calibrated in the score units they are compared against, "
        "and in the starting range of the `escalation_threshold` trait.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Calibrating escalation thresholds to the compared distribution", preamble
    )


def _threshold_range(text: str) -> tuple[float, float]:
    low, _, high = text.partition(":")
    return (float(low), float(high))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--threshold-ranges",
        type=_threshold_range,
        nargs="*",
        default=[(0.1, 0.5), (0.05, 0.3)],
        help="Extra `low:high` starting ranges for the escalation_threshold trait.",
    )
    parser.add_argument(
        "--correct-report-value",
        type=float,
        default=8.0,
        help="Fixed correct_report_attention_value supplying the value of a correct report.",
    )
    parser.add_argument(
        "--break-even-precision",
        type=float,
        default=0.2,
        help="Fixed false-alarm pricing target from the previous lever.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Print the report without rewriting the committed docs/ artifacts. "
            "Writing expects the repository root as the working directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the calibration sweep and write the JSON and markdown artifacts."""
    args = _parse_args(argv)
    results = run_sweep(args)
    if not args.no_write:
        with open(_JSON_ARTIFACT, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        with open(_REPORT_ARTIFACT, "w", encoding="utf-8") as handle:
            handle.write(markdown_report(results))
    print(markdown_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
