#!/usr/bin/env python3
"""Measure whether report correctness pays, link by link, on a real instrument.

The ceiling harness answers "how often are reports correct?". This script answers
the prior question: does being correct change an agent's currency balances and its
offspring count at all? For each arm it runs the same fixed initial parameters and
records, per agent, correctness alongside every currency inflow, then reports the
correlation of each link in the chain

    correctness -> user trust -> attention income -> reproduction
    (information yield / peer subsidy -> reproduction, as the competing path)

so the severed link is identifiable rather than inferred. The oracle-invasion arm
supplies agents that are correct by construction, giving the correctness axis
enough spread to measure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import measurement_support
import numpy as np

from tattletots.telemetry.payoff_ledger import PayoffLedger

_JSON_ARTIFACT = "docs/payoff-coupling.json"
_REPORT_ARTIFACT = "docs/payoff-coupling.md"

harness = measurement_support.load_harness()


def measure_run(
    arm: str,
    grounded_fraction: float,
    seed: int,
    options: Any,
    attention_budget_scale: float = 1.0,
) -> dict[str, Any]:
    """Run one arm/seed cell with a payoff ledger attached."""
    point = harness.GridPoint(
        arm=arm,
        grounded_fraction=grounded_fraction,
        grounded_multiplier=1.0,
    )
    adapter = harness.build_adapter(options.adapter_spec, seed, options.steps)
    world = harness.build_world(adapter, point, seed, options)
    for user in world.users.values():
        user.attention_budget *= attention_budget_scale
    ledger = PayoffLedger()
    measurement_support.drive_world(harness, world, adapter, options.steps, ledger)

    summary = world.telemetry.summary()
    return {
        "arm": arm,
        "grounded_input_fraction": grounded_fraction,
        "attention_budget_scale": attention_budget_scale,
        "false_alarm_penalty": world.config.false_alarm_penalty,
        "seed": seed,
        "correct_report_rate": float(summary["precision"]),
        "grounded_yield_share": float(summary["grounded_yield_share"]),
        "final_population": int(summary["final_population"]),
        "coupling": ledger.coupling_summary(),
    }


def _pooled(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each coupling statistic across seeds for one cell."""
    couplings = [run["coupling"] for run in runs if run["coupling"].get("n_adults", 0) > 0]
    if not couplings:
        return {"n_runs": len(runs)}
    scalar_keys = [
        key
        for key, value in couplings[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary: dict[str, Any] = {
        f"mean_{key}": _pooled([float(coupling[key]) for coupling in couplings])
        for key in scalar_keys
    }
    gate_keys = couplings[0]["reproduction_gate"].keys()
    summary["reproduction_gate"] = {
        key: _pooled([float(coupling["reproduction_gate"][key]) for coupling in couplings])
        for key in gate_keys
    }
    summary["mean_correct_report_rate"] = _pooled(
        [float(run["correct_report_rate"]) for run in runs]
    )
    summary["mean_grounded_yield_share"] = _pooled(
        [float(run["grounded_yield_share"]) for run in runs]
    )
    summary["mean_final_population"] = _pooled([float(run["final_population"]) for run in runs])
    summary["n_runs"] = len(runs)
    return summary


def run_measurement(
    options: Any,
    arms: Sequence[str],
    grounded_fraction: float,
    seeds: Sequence[int],
    attention_budget_scale: float = 1.0,
) -> dict[str, Any]:
    """Run every arm across every seed at one fixed grounded-access setting."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        runs[arm] = [
            measure_run(arm, grounded_fraction, seed, options, attention_budget_scale)
            for seed in seeds
        ]
    return {
        "adapter": options.adapter_spec,
        "steps": options.steps,
        "seeds": list(seeds),
        "grounded_input_fraction": grounded_fraction,
        "attention_budget_scale": attention_budget_scale,
        "false_alarm_penalty": options.extra_config.get("false_alarm_penalty"),
        "config_overrides": {
            key: value
            for key, value in options.extra_config.items()
            if key != "false_alarm_penalty"
        },
        "runs": runs,
        "summary": {arm: _summarize(cell) for arm, cell in runs.items()},
    }


_LINKS = (
    ("correctness -> user trust", "mean_corr_correct_reports_trust"),
    ("precision -> user trust", "mean_corr_precision_trust"),
    ("trust -> attention income", "mean_corr_trust_attention_income"),
    ("correctness -> attention income", "mean_corr_correct_reports_attention_income"),
    ("correctness -> information income", "mean_corr_correct_reports_information_income"),
    ("attention income -> offspring", "mean_corr_attention_income_offspring"),
    ("information income -> offspring", "mean_corr_information_income_offspring"),
    ("correctness -> offspring", "mean_corr_correct_reports_offspring"),
    ("report volume -> attention income", "mean_corr_reports_issued_attention_income"),
    ("report volume -> offspring", "mean_corr_reports_issued_offspring"),
)


def markdown_report(results: dict[str, Any]) -> str:
    """Render the per-link coupling table and the currency-scale comparison."""
    lines = [
        "# Does correctness pay? Per-link payoff coupling",
        "",
        f"- Adapter: `{results['adapter']}`",
        f"- Steps per run: `{results['steps']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in results['seeds'])}`",
        f"- Grounded input fraction (fixed): `{results['grounded_input_fraction']:g}`",
        f"- User attention-budget scale: `{results['attention_budget_scale']:g}`",
        f"- False-alarm penalty override: `{results['false_alarm_penalty']}`",
        f"- Config overrides: `{results['config_overrides'] or 'none'}`",
        "",
        "## Coupling of each link (Pearson r, mean over seeds)",
        "",
        "| Link | " + " | ".join(f"`{arm}`" for arm in results["summary"]) + " |",
        "|---" * (len(results["summary"]) + 1) + "|",
    ]
    for label, key in _LINKS:
        cells = " | ".join(
            f"{cell.get(key, 0.0):+.3f}" if cell.get("n_runs") else "n/a"
            for cell in results["summary"].values()
        )
        lines.append(f"| {label} | {cells} |")

    lines.extend(["", "## Currency scale and rationing", ""])
    scale_rows = (
        ("Correct-report rate", "mean_correct_report_rate", "{:.2%}"),
        ("Attention income / agent-step", "mean_mean_attention_income_per_step", "{:.4f}"),
        ("Information income / agent-step", "mean_mean_information_income_per_step", "{:.4f}"),
        ("Information share of reserves", "mean_mean_information_share_of_reserves", "{:.2%}"),
        (
            "Peer-subsidy share of info income",
            "mean_mean_subsidy_share_of_information_income",
            "{:.2%}",
        ),
        ("Mean offspring, ever-correct agents", "mean_correct_group_mean_offspring", "{:.2f}"),
        (
            "Mean offspring, never-correct agents",
            "mean_never_correct_group_mean_offspring",
            "{:.2f}",
        ),
        ("Silent adults (never reported)", "mean_n_silent_adults", "{:.1f}"),
        ("Attention income, silent adults", "mean_silent_mean_attention_income", "{:.4f}"),
        ("Attention income, reporting adults", "mean_reporting_mean_attention_income", "{:.4f}"),
        ("Mean offspring, silent adults", "mean_silent_mean_offspring", "{:.2f}"),
        ("Mean offspring, reporting adults", "mean_reporting_mean_offspring", "{:.2f}"),
        ("Trust break-even precision", "mean_trust_break_even_precision", "{:.2%}"),
        (
            "False alarm cost (agent-steps of attention income)",
            "mean_false_alarm_penalty_in_attention_income_steps",
            "{:.1f}",
        ),
    )
    falsification_rows = (
        ("Generations with reports", "mean_generations_observed", "{:.1f}"),
        (
            "Correct-report rate slope per generation",
            "mean_precision_generation_slope",
            "{:+.4f}",
        ),
        ("Parent-child offspring correlation", "mean_corr_parent_child_offspring", "{:+.3f}"),
        ("Parent-child pairs", "mean_n_parent_child_pairs", "{:.1f}"),
        (
            "Parent-child precision correlation",
            "mean_corr_parent_child_precision",
            "{:+.3f}",
        ),
    )
    lines.append("| Quantity | " + " | ".join(f"`{arm}`" for arm in results["summary"]) + " |")
    lines.append("|---" * (len(results["summary"]) + 1) + "|")
    for label, key, fmt in scale_rows:
        cells = " | ".join(
            fmt.format(cell.get(key, 0.0)) if cell.get("n_runs") else "n/a"
            for cell in results["summary"].values()
        )
        lines.append(f"| {label} | {cells} |")

    lines.extend(["", "### Falsification clauses, measured on these runs", ""])
    lines.append("| Quantity | " + " | ".join(f"`{arm}`" for arm in results["summary"]) + " |")
    lines.append("|---" * (len(results["summary"]) + 1) + "|")
    for label, key, fmt in falsification_rows:
        cells = " | ".join(
            fmt.format(cell.get(key, 0.0)) if cell.get("n_runs") else "n/a"
            for cell in results["summary"].values()
        )
        lines.append(f"| {label} | {cells} |")

    lines.extend(["", "### Reproduction gating (share of agent-steps)", ""])
    gate_keys = (
        "eligible_share",
        "co_limited_share",
        "attention_limited_share",
        "information_limited_share",
        "population_capped_step_share",
    )
    lines.append("| Condition | " + " | ".join(f"`{arm}`" for arm in results["summary"]) + " |")
    lines.append("|---" * (len(results["summary"]) + 1) + "|")
    for key in gate_keys:
        cells = " | ".join(
            f"{cell.get('reproduction_gate', {}).get(key, 0.0):.2%}"
            if cell.get("n_runs")
            else "n/a"
            for cell in results["summary"].values()
        )
        lines.append(f"| {key} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def _parse_config_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse KEY=VALUE config overrides into typed SimulationConfig values."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not key or not raw:
            raise ValueError(f"config override must be KEY=VALUE: {pair}")
        lowered = raw.lower()
        value: Any
        if lowered in ("true", "false"):
            value = lowered == "true"
        else:
            value = float(raw)
        overrides[key] = value
    return overrides


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arms", nargs="+", default=["ordinary", "oracle_invasion"])
    parser.add_argument(
        "--attention-budget-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplier on every user's attention budget.",
    )
    parser.add_argument(
        "--false-alarm-penalty",
        type=float,
        default=None,
        help="Diagnostic override of SimulationConfig.false_alarm_penalty.",
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional SimulationConfig overrides, e.g. reproduction_merit_ordering=true.",
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


def _extra_config(args: argparse.Namespace) -> dict[str, Any]:
    extra = _parse_config_overrides(args.config)
    if args.false_alarm_penalty is not None:
        extra["false_alarm_penalty"] = args.false_alarm_penalty
    return extra


def main(argv: Sequence[str] | None = None) -> int:
    """Run the payoff-coupling measurement and write the JSON and markdown output."""
    args = _parse_args(argv)
    options = harness.HarnessOptions(
        adapter_spec=args.adapter,
        steps=args.steps,
        seeds=tuple(args.seeds),
        grounded_fractions=(args.grounded_fraction,),
        initial_population=args.initial_population,
        max_population=args.max_population,
        arms=tuple(args.arms),
        extra_config=_extra_config(args),
    )
    results = run_measurement(
        options,
        args.arms,
        args.grounded_fraction,
        args.seeds,
        args.attention_budget_scale,
    )
    if not args.no_write:
        with open(_JSON_ARTIFACT, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        with open(_REPORT_ARTIFACT, "w", encoding="utf-8") as handle:
            handle.write(markdown_report(results))
    print(markdown_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
