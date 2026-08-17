#!/usr/bin/env python3
"""Measure whether reproductive excess is what limits differential reproduction.

`docs/response-gate-measurement.md` keyed reproduction on rank in verified correctness and
cleared clause 1, but clause 2 stayed at 0/40 for a measured reason: about two thirds of
adults are eligible on a given step and the population cap binds on about one third, so an
ordering over parents mostly decides who reproduces *first*, moving lineage output by about
0.03 offspring. Ranking is only consequential when eligible parents outnumber the
recruitment the environment can afford.

This script sweeps `SimulationConfig.reproduction_recruitment_share`, the share of eligible
parents allowed to recruit an offspring on a step, with every earlier lever held at its
measured setting, and reports the reproductive excess and the opportunity for selection
`I = var(offspring) / mean(offspring)^2` alongside both falsification clauses. The share is
a limit on recruitment, not a payment, so it stays on the correct side of the no-scaffolding
constraint.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/reproductive-excess.json"
_REPORT_ARTIFACT = "docs/reproductive-excess.md"

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def excess_config(share: float, args: argparse.Namespace) -> dict[str, Any]:
    """Config overrides for one recruitment-share arm, holding every lever fixed."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": args.correct_report_value,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": True,
        "reproduction_correctness_weight": args.correctness_weight,
        "reproduction_recruitment_share": share,
        "gene_pool": {"escalation_threshold_range": list(args.threshold_range)},
    }
    if args.break_even_precision is not None:
        extra["false_alarm_break_even_precision"] = args.break_even_precision
    return extra


def arm_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """The unlimited-recruitment control, then one arm per recruitment share."""
    shares = sorted({1.0, *args.shares}, reverse=True)
    return [(f"recruitment_share_{share:g}", excess_config(share, args)) for share in shares]


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every recruitment-share arm at the same fixed initial parameters."""
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs(args))
    results["correct_report_attention_value"] = args.correct_report_value
    results["break_even_precision"] = args.break_even_precision
    results["correctness_weight"] = args.correctness_weight
    results["shares"] = sorted({1.0, *args.shares}, reverse=True)
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the sweep as one table of recruitment shares against clause metrics."""
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
        "",
        "Every arm shares the same initial parameters; the arms differ only in "
        "`reproduction_recruitment_share`, the share of the step's eligible parents that "
        "may recruit an offspring. The `recruitment_share_1` arm is the unlimited "
        "recruitment every earlier lever was measured under, where the population cap is "
        "the only limit.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Reproductive excess and the opportunity for selection", preamble
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--shares",
        type=float,
        nargs="+",
        default=[0.5, 0.25, 0.1],
        help="Recruitment shares to compare; 1.0 is always included as the control.",
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=1.0,
        help="Fixed response-gate weight from the previous lever.",
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
    """Run the reproductive-excess sweep and write the JSON and markdown artifacts."""
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
