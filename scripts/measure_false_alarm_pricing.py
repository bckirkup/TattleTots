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
from pathlib import Path
from typing import Any

import measurement_support

_JSON_ARTIFACT = "docs/false-alarm-pricing.json"
_REPORT_ARTIFACT = "docs/false-alarm-pricing.md"

harness = measurement_support.load_harness()
coupling = measurement_support.load_module(
    "measure_payoff_coupling",
    Path(__file__).resolve().parent / "measure_payoff_coupling.py",
)

_ROWS = (
    ("Realized break-even precision", "mean_realized_break_even_precision", "{:.2%}"),
    ("Attention charged per false alarm", "mean_mean_false_alarm_price", "{:.4f}"),
    ("Attention value of a correct report", "mean_mean_correct_report_value", "{:.4f}"),
    ("Correct-report rate", "mean_correct_report_rate", "{:.2%}"),
    ("Reports per adult lifetime", "mean_mean_reports_per_adult", "{:.2f}"),
    ("Adult steps per agent", "mean_mean_adult_steps", "{:.2f}"),
    ("Silent-adult share", "mean_silent_adult_share", "{:.2%}"),
    ("Attention income / agent-step", "mean_mean_attention_income_per_step", "{:.4f}"),
    (
        "Fitness alignment b (correctness -> offspring)",
        "mean_corr_correct_reports_offspring",
        "{:+.3f}",
    ),
    ("Clause 1: correct-report slope / generation", "mean_precision_generation_slope", "{:+.4f}"),
    ("Clause 2: parent-child offspring correlation", "mean_corr_parent_child_offspring", "{:+.3f}"),
    ("Parent-child precision correlation", "mean_corr_parent_child_precision", "{:+.3f}"),
    ("Parent-child pairs", "mean_n_parent_child_pairs", "{:.1f}"),
)


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
    arms: dict[str, Any] = {}
    for target in [None, *args.targets]:
        label = "flat_penalty" if target is None else f"break_even_{target:g}"
        options = harness.HarnessOptions(
            adapter_spec=args.adapter,
            steps=args.steps,
            seeds=tuple(args.seeds),
            grounded_fractions=(args.grounded_fraction,),
            initial_population=args.initial_population,
            max_population=args.max_population,
            arms=(args.arm,),
            extra_config=pricing_config(target, args.correct_report_value),
        )
        results = coupling.run_measurement(
            options,
            (args.arm,),
            args.grounded_fraction,
            args.seeds,
        )
        arms[label] = {
            "break_even_target": target,
            "summary": results["summary"][args.arm],
            "runs": results["runs"][args.arm],
        }
    return {
        "adapter": args.adapter,
        "steps": args.steps,
        "seeds": list(args.seeds),
        "arm": args.arm,
        "grounded_input_fraction": args.grounded_fraction,
        "max_population": args.max_population,
        "correct_report_attention_value": args.correct_report_value,
        "arms": arms,
    }


def markdown_report(results: dict[str, Any]) -> str:
    """Render the pricing sweep as one table of arms against reporting economics."""
    arms = results["arms"]
    lines = [
        "# Repricing false alarms against reachable precision",
        "",
        f"- Adapter: `{results['adapter']}`",
        f"- Arm: `{results['arm']}`",
        f"- Steps per run: `{results['steps']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in results['seeds'])}`",
        f"- Grounded input fraction (fixed): `{results['grounded_input_fraction']:g}`",
        f"- Max population (fixed): `{results['max_population']}`",
        (
            "- `correct_report_attention_value` (fixed across arms): "
            f"`{results['correct_report_attention_value']:g}`"
        ),
        "",
        "Every arm shares the same initial parameters; only "
        "`false_alarm_break_even_precision` differs.",
        "",
        "| Quantity | " + " | ".join(f"`{name}`" for name in arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    for label, key, fmt in _ROWS:
        cells = " | ".join(
            fmt.format(arm["summary"].get(key, 0.0)) if arm["summary"].get("n_runs") else "n/a"
            for arm in arms.values()
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    return "\n".join(lines)


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
