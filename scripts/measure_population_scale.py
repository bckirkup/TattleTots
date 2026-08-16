#!/usr/bin/env python3
"""Measure how the falsification clauses behave as effective population rises.

`docs/domain-richness-requirement.md` derives a required breeding population that scales
as `1/b**2`: ~107 agents at perfect fitness alignment and ~555 at the alignment actually
measured. Levers 1 and 2 exhausted the sample-size term (`k` is now ~9 reports per adult
lifetime) without moving `b`, so this lever asks the remaining question the requirement
poses: does the drift term matter at this scale, i.e. do the clause metrics improve when
the population cap stops being 60?

Each arm holds every other initial parameter fixed, including the two earlier levers, and
varies only the population cap and the founding population it is seeded with.

Raising the cap alone confounds two things, because the users' attention budget is a fixed
total: at a larger cap the same budget is split more ways, so per-capita solvency, and with
it adult lifespan and reports per lifetime, falls as the population rises. The
`_per_capita` arms therefore scale the attention budget with the cap, holding per-capita
solvency at the value the earlier levers were measured at, so the drift term is measured on
its own rather than against a shrinking per-agent income.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import config_arm_sweep
import measurement_support

_JSON_ARTIFACT = "docs/population-scale.json"
_REPORT_ARTIFACT = "docs/population-scale.md"
_FOUNDER_SHARE = 1.0 / 3.0

harness = measurement_support.load_harness()
coupling = config_arm_sweep.load_coupling()


def founding_population(cap: int) -> int:
    """Seed each arm with the same share of its cap, so founding diversity scales too."""
    return max(2, round(cap * _FOUNDER_SHARE))


def population_config(
    cap: int, args: argparse.Namespace, *, per_capita_attention: bool = False
) -> dict[str, Any]:
    """Config overrides for one population arm, holding both earlier levers fixed."""
    extra: dict[str, Any] = {
        "correct_report_attention_value": args.correct_report_value,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": True,
        "max_population": cap,
        "initial_population": founding_population(cap),
        "gene_pool": {"escalation_threshold_range": list(args.threshold_range)},
    }
    if args.break_even_precision is not None:
        extra["false_alarm_break_even_precision"] = args.break_even_precision
    if per_capita_attention:
        extra["attention_budget_scale"] = cap / args.reference_cap
    return extra


def arm_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """One fixed-budget arm per cap, then the same caps at constant per-capita attention."""
    caps = sorted(args.caps)
    configs = [(f"cap_{cap}", population_config(cap, args)) for cap in caps]
    configs.extend(
        (f"cap_{cap}_per_capita", population_config(cap, args, per_capita_attention=True))
        for cap in caps
        if cap != args.reference_cap
    )
    return configs


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run every population arm at the same fixed initial parameters."""
    results = config_arm_sweep.run_arms(harness, coupling, args, arm_configs(args))
    results["correct_report_attention_value"] = args.correct_report_value
    results["break_even_precision"] = args.break_even_precision
    results["caps"] = sorted(args.caps)
    results["reference_cap"] = args.reference_cap
    return results


def markdown_report(results: dict[str, Any]) -> str:
    """Render the population sweep as one table of caps against clause metrics."""
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
        f"- Reference cap the earlier levers were measured at: `{results['reference_cap']}`",
        "",
        "Every arm shares the same initial parameters except the population cap and the "
        "founding population, which is a fixed share of the cap. The `Max population` line "
        "above is the sweep default; each arm's own cap is its column label. The "
        "`_per_capita` arms additionally scale the users' attention budget with the cap, so "
        "per-capita solvency stays at its reference value instead of falling as the "
        "population grows.",
    ]
    return config_arm_sweep.markdown_report(
        results, "Population scale against the falsification clauses", preamble
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arm", default="ordinary")
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=[60, 125, 250],
        help="Population caps to compare; 60 is the cap every earlier lever was measured at.",
    )
    parser.add_argument(
        "--reference-cap",
        type=int,
        default=60,
        help=(
            "Cap whose per-capita attention budget the `_per_capita` arms preserve, and which "
            "therefore needs no per-capita arm of its own."
        ),
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=(0.05, 0.3),
        metavar=("LOW", "HIGH"),
        help="Starting range of the escalation_threshold trait, fixed from the previous lever.",
    )
    measurement_support.add_lever_arguments(
        parser, break_even_help="Fixed false-alarm pricing target from the first lever."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the population sweep and write the JSON and markdown artifacts."""
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
