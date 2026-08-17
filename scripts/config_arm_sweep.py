"""Shared sweep for measurements that vary one config knob at fixed initial parameters.

Each lever in `docs/domain-richness-requirement.md` is measured the same way: run the same
arm and seeds under a handful of config settings, then read the reporting economics and
both falsification clauses off the payoff ledger. The per-arm running and the table
rendering live here so each lever's script contains only the settings it varies.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import measurement_support

ROWS: tuple[tuple[str, str, str], ...] = (
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
    ("Reproductive excess (eligible / slot)", "mean_reproductive_excess", "{:.2f}"),
    ("Slot-limited step share", "mean_slot_limited_step_share", "{:.2%}"),
    ("Opportunity for selection I", "mean_opportunity_for_selection", "{:.3f}"),
    ("Mean offspring per adult", "mean_mean_offspring", "{:.2f}"),
    ("Died before the run ended (adults)", "mean_died_adult_share", "{:.2%}"),
    ("Childless adult share", "mean_childless_adult_share", "{:.2%}"),
    ("Rank persistence (early vs late life)", "mean_rank_persistence", "{:+.3f}"),
    ("Rank -> adult lifespan", "mean_corr_rank_adult_steps", "{:+.3f}"),
    ("Rank -> offspring", "mean_corr_rank_offspring", "{:+.3f}"),
    ("Lifespan -> offspring", "mean_corr_adult_steps_offspring", "{:+.3f}"),
    ("Clause 1: correct-report slope / generation", "mean_precision_generation_slope", "{:+.4f}"),
    ("Clause 2: parent-child offspring correlation", "mean_corr_parent_child_offspring", "{:+.3f}"),
    ("Parent-child precision correlation", "mean_corr_parent_child_precision", "{:+.3f}"),
    ("Parent-child pairs", "mean_n_parent_child_pairs", "{:.1f}"),
    ("Adults scored", "mean_n_adults", "{:.1f}"),
    ("Final population", "mean_final_population", "{:.1f}"),
)

_NON_CONFIG_KEYS = frozenset({"gene_pool", "attention_budget_scale"})


def load_coupling() -> ModuleType:
    """Load the payoff-coupling measurement for its ledgered run/summary helpers."""
    return measurement_support.load_module(
        "measure_payoff_coupling",
        Path(__file__).resolve().parent / "measure_payoff_coupling.py",
    )


def run_arms(
    harness: ModuleType,
    coupling: ModuleType,
    args: argparse.Namespace,
    arm_configs: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Run each labelled config setting over the same arm and seeds.

    An arm's config may carry keys that are not `SimulationConfig` fields and are applied
    to the run separately: `gene_pool` sets the initial trait distribution, and
    `attention_budget_scale` multiplies every user's attention budget.
    """
    arms: dict[str, Any] = {}
    for label, arm_config in arm_configs:
        extra_config = {
            key: value for key, value in arm_config.items() if key not in _NON_CONFIG_KEYS
        }
        options = harness.HarnessOptions(
            adapter_spec=args.adapter,
            steps=args.steps,
            seeds=tuple(args.seeds),
            grounded_fractions=(args.grounded_fraction,),
            initial_population=args.initial_population,
            max_population=args.max_population,
            arms=(args.arm,),
            extra_config=extra_config,
            gene_pool=arm_config.get("gene_pool"),
        )
        results = coupling.run_measurement(
            options,
            (args.arm,),
            args.grounded_fraction,
            args.seeds,
            attention_budget_scale=float(arm_config.get("attention_budget_scale", 1.0)),
        )
        arms[label] = {
            "config": arm_config,
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
        "arms": arms,
    }


def markdown_report(
    results: dict[str, Any],
    title: str,
    preamble: Sequence[str] = (),
) -> str:
    """Render one arm-per-column table of reporting economics and clause metrics."""
    arms = results["arms"]
    lines = [
        f"# {title}",
        "",
        f"- Adapter: `{results['adapter']}`",
        f"- Arm: `{results['arm']}`",
        f"- Steps per run: `{results['steps']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in results['seeds'])}`",
        f"- Grounded input fraction (fixed): `{results['grounded_input_fraction']:g}`",
        f"- Max population (fixed): `{results['max_population']}`",
        *preamble,
        "",
        "| Quantity | " + " | ".join(f"`{name}`" for name in arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    for label, key, fmt in ROWS:
        cells = " | ".join(
            fmt.format(arm["summary"].get(key, 0.0)) if arm["summary"].get("n_runs") else "n/a"
            for arm in arms.values()
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    return "\n".join(lines)
