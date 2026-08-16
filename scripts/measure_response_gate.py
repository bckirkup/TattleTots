#!/usr/bin/env python3
"""Measure what rationing reproduction by verified correctness buys.

Levers 1, 2 and 4 (`docs/false-alarm-pricing-measurement.md`,
`docs/threshold-calibration-measurement.md`, `docs/population-scale.md`) each moved the
quantity they targeted and left fitness alignment `b` at +0.02..+0.07. The remaining
suspect is the response gate itself: reproduction is rationed by reserve sufficiency, and
reserves are dominated by correctness-blind information income, so an agent that reports
correctly is not ordered ahead of one that merely accumulated.

This script sweeps `SimulationConfig.reproduction_correctness_weight`, which mixes rank in
verified correctness into that ordering, at otherwise fixed initial parameters — every
earlier lever held at its measured setting — and reports reporting economics alongside both
falsification clauses so this lever stays separable from the ones before it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/response-gate.json"
_REPORT_ARTIFACT = "docs/response-gate.md"

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def gate_config(weight: float, args: argparse.Namespace) -> dict[str, Any]:
    """Config overrides for one gate arm, holding every earlier lever fixed."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": args.correct_report_value,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": True,
        "reproduction_correctness_weight": weight,
        "gene_pool": {"escalation_threshold_range": list(args.threshold_range)},
    }
    if args.break_even_precision is not None:
        extra["false_alarm_break_even_precision"] = args.break_even_precision
    return extra


def arm_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """The reserves-only control, then one arm per correctness weight."""
    weights = sorted({0.0, *args.weights})
    return [(f"correctness_weight_{weight:g}", gate_config(weight, args)) for weight in weights]


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every gate arm at the same fixed initial parameters."""
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs(args))
    results["correct_report_attention_value"] = args.correct_report_value
    results["break_even_precision"] = args.break_even_precision
    results["weights"] = sorted({0.0, *args.weights})
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the gate sweep as one table of correctness weights against clause metrics."""
    preamble = [
        (
            "- `correct_report_attention_value` (fixed across arms): "
            f"`{results['correct_report_attention_value']:g}`"
        ),
        (
            "- `false_alarm_break_even_precision` (fixed across arms): "
            f"`{results['break_even_precision']}`"
        ),
        "- `escalation_calibration_in_score_units` (fixed across arms): `True`",
        "- `reproduction_merit_ordering` (fixed across arms): `True`",
        "",
        "Every arm shares the same initial parameters; the arms differ only in "
        "`reproduction_correctness_weight`, the share of reproductive merit carried by "
        "rank in verified correctness rather than rank in reserve sufficiency. The "
        "`correctness_weight_0` arm is the reserves-only ordering every earlier lever was "
        "measured under.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Rationing reproduction by verified correctness", preamble
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Correctness weights to compare; 0.0 is always included as the control.",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.05, 0.3],
        metavar=("LOW", "HIGH"),
        help="Fixed starting escalation_threshold range from the calibration lever.",
    )
    measurement_support.add_lever_arguments(
        parser, break_even_help="Fixed false-alarm pricing target from the first lever."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the response-gate sweep and write the JSON and markdown artifacts."""
    args = _parse_args(argv)
    results = run_sweep(args)
    measurement_support.emit_artifacts(
        results,
        markdown_report(results),
        json_path=_JSON_ARTIFACT,
        report_path=_REPORT_ARTIFACT,
        write=not args.no_write,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
