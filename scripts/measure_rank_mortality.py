#!/usr/bin/env python3
"""Measure whether reproductive rank is coupled to mortality.

`docs/reproductive-excess-measurement.md` refuted per-step scarcity as the missing term for
clause 2: eligible parents already outnumber affordable recruits by about 32:1, yet every
adult ends up with 1.26-1.28 offspring, because eligibility is not consumed and an adult
lives about 52 adult steps, so losing a recruitment contest only delays a turn that arrives
anyway. Ranking can only produce differential lineage output if death reaches a low-ranked
adult while it is still waiting.

This script measures that coupling directly. It reports, per arm, whether an adult's rank in
verified correctness persists across its life (early-life rank against late-life rank),
whether rank predicts adult lifespan, and whether the adults that die childless are the
low-ranked ones. The arms vary the scarcity of user attention, which is the currency whose
insolvency causes essentially every death and which correct reports are what earn: a scarcer
budget is a harsher environment, not a payment to anyone, so it stays on the correct side of
the no-scaffolding constraint.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/rank-mortality.json"
_REPORT_ARTIFACT = "docs/rank-mortality.md"

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def mortality_config(budget_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    """Config overrides for one attention-scarcity arm, holding every lever fixed."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": args.correct_report_value,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": True,
        "reproduction_correctness_weight": args.correctness_weight,
        "reproduction_recruitment_share": args.recruitment_share,
        "attention_budget_scale": budget_scale,
        "gene_pool": {"escalation_threshold_range": list(args.threshold_range)},
    }
    if args.break_even_precision is not None:
        extra["false_alarm_break_even_precision"] = args.break_even_precision
    return extra


def arm_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """The unscaled-budget control, then one arm per attention-budget scale."""
    scales = sorted({1.0, *args.budget_scales}, reverse=True)
    return [(f"budget_scale_{scale:g}", mortality_config(scale, args)) for scale in scales]


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every attention-scarcity arm at the same fixed initial parameters."""
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs(args))
    results["correct_report_attention_value"] = args.correct_report_value
    results["break_even_precision"] = args.break_even_precision
    results["correctness_weight"] = args.correctness_weight
    results["recruitment_share"] = args.recruitment_share
    results["budget_scales"] = sorted({1.0, *args.budget_scales}, reverse=True)
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the sweep as one table of attention scarcity against clause metrics."""
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
        (
            "- `reproduction_correctness_weight` (fixed across arms): "
            f"`{results['correctness_weight']:g}`"
        ),
        (
            "- `reproduction_recruitment_share` (fixed across arms): "
            f"`{results['recruitment_share']:g}`"
        ),
        "",
        "Every arm shares the same initial parameters; the arms differ only in the scale "
        "applied to each user's attention budget, the currency whose insolvency causes "
        "essentially every death. The `budget_scale_1` arm is the environment every "
        "earlier lever was measured under.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Rank-coupled mortality: does a low-ranked adult die while waiting?", preamble
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--budget-scales",
        type=float,
        nargs="+",
        default=[0.5, 0.25, 0.1],
        help="Attention-budget scales to compare; 1.0 is always included as the control.",
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=1.0,
        help="Fixed response-gate weight from the previous lever.",
    )
    parser.add_argument(
        "--recruitment-share",
        type=float,
        default=1.0,
        help="Fixed recruitment share from the reproductive-excess lever.",
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
    """Run the rank-mortality sweep and write the JSON and markdown artifacts."""
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
