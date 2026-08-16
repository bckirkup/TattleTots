#!/usr/bin/env python3
"""Measure what repricing false alarms against reachable precision buys.

`docs/reporting-opportunity-measurement.md` found that reporting is priced against a
precision no evolved agent can reach: a flat false-alarm penalty against a small
correctness-sensitive attention income puts break-even precision near 80% while the
instrument's decoder ceiling is far below it, so silence is the evolved optimum and the
fitness alignment `b` in `docs/domain-richness-requirement.md` stays low.

This script sweeps `SimulationConfig.false_alarm_break_even_precision` at otherwise
fixed initial parameters and reports, per arm, the realized break-even precision an
agent faced, how much it then reported, and both falsification clauses, so the effect
of pricing alone is separable from the later levers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/false-alarm-pricing.json"
_REPORT_ARTIFACT = "docs/false-alarm-pricing.md"

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def pricing_config(target: float | None, correct_report_value: float) -> dict[str, Any]:
    """Config overrides for one pricing arm; only the pricing target varies."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": correct_report_value,
        "reproduction_merit_ordering": True,
    }
    if target is not None:
        extra["false_alarm_break_even_precision"] = target
    return extra


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every pricing arm at the same fixed initial parameters."""
    arm_configs = [
        (
            "flat_penalty" if target is None else f"break_even_{target:g}",
            pricing_config(target, args.correct_report_value),
        )
        for target in [None, *args.targets]
    ]
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs)
    results["correct_report_attention_value"] = args.correct_report_value
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the pricing sweep as one table of arms against reporting economics."""
    preamble = [
        (
            "- `correct_report_attention_value` (fixed across arms): "
            f"`{results['correct_report_attention_value']:g}`"
        ),
        "",
        "Every arm shares the same initial parameters; only "
        "`false_alarm_break_even_precision` differs.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Repricing false alarms against reachable precision", preamble
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.4, 0.2, 0.1, 0.05],
        help="Break-even precision targets to price against, in addition to the flat penalty.",
    )
    parser.add_argument(
        "--correct-report-value",
        type=float,
        default=8.0,
        help="Fixed correct_report_attention_value supplying the value of a correct report.",
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
    """Run the pricing sweep and write the JSON and markdown artifacts."""
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
