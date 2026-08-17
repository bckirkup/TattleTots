#!/usr/bin/env python3
"""Per-seed reliability of the two falsification clauses under the payoff fixes.

The arm-averaged coupling report hides whether a clause is *reliably* met: a mean
generational slope of +0.001 can come from every seed rising slightly or from a
few seeds rising while the rest fall. This script runs the ordinary (evolved) arm
over many seeds at several doses of `correct_report_attention_value` and reports,
per dose, how many seeds actually satisfy each clause.

Prints only; it writes no artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEASURE_PATH = _REPO_ROOT / "scripts" / "measure_payoff_coupling.py"
_CORRELATION_CLAUSE = 0.2


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dose_options(measure: ModuleType, value: float, args: argparse.Namespace) -> Any:
    extra_config: dict[str, Any] = {
        "correct_report_attention_value": value,
        "reproduction_merit_ordering": args.merit_ordering,
    }
    if args.break_even_precision is not None:
        extra_config["false_alarm_break_even_precision"] = args.break_even_precision
    if args.score_units:
        extra_config["escalation_calibration_in_score_units"] = True
    if args.correctness_weight > 0.0:
        extra_config["reproduction_correctness_weight"] = args.correctness_weight
    if args.recruitment_share < 1.0:
        extra_config["reproduction_recruitment_share"] = args.recruitment_share
    gene_pool = (
        {"escalation_threshold_range": list(args.threshold_range)}
        if args.threshold_range is not None
        else None
    )
    return measure.harness.HarnessOptions(
        adapter_spec=args.adapter,
        steps=args.steps,
        seeds=tuple(args.seeds),
        initial_population=args.initial_population,
        max_population=args.max_population,
        extra_config=extra_config,
        gene_pool=gene_pool,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", default="tattletots.scenarios.sparse_sensor:SparseSensorScenario"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    parser.add_argument("--doses", type=float, nargs="+", default=[0.0, 32.0, 128.0])
    parser.add_argument("--grounded-fraction", type=float, default=0.67)
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    parser.add_argument("--merit-ordering", action="store_true", default=True)
    parser.add_argument("--no-merit-ordering", dest="merit_ordering", action="store_false")
    parser.add_argument(
        "--break-even-precision",
        type=float,
        default=None,
        help=(
            "Price false alarms so reporting breaks even at this precision instead of "
            "at the flat penalty."
        ),
    )
    parser.add_argument(
        "--score-units",
        action="store_true",
        help=(
            "Calibrate adaptive escalation thresholds in the normalized score units they "
            "are compared against, instead of the raw anomaly window."
        ),
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=0.0,
        help=(
            "Share of reproductive merit carried by rank in verified correctness rather "
            "than rank in reserve sufficiency."
        ),
    )
    parser.add_argument(
        "--recruitment-share",
        type=float,
        default=1.0,
        help=(
            "Share of the step's eligible parents allowed to recruit an offspring, the "
            "reproductive excess an ordering over parents can act on."
        ),
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Starting range of the escalation_threshold trait in the gene pool.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Report per-seed clause satisfaction for each attention-value dose."""
    args = _parse_args(argv)
    measure = _load("measure_payoff_coupling", _MEASURE_PATH)
    for value in args.doses:
        options = _dose_options(measure, value, args)
        slopes: list[float] = []
        offspring_corrs: list[float] = []
        precision_corrs: list[float] = []
        rates: list[float] = []
        reports_per_adult: list[float] = []
        excess: list[float] = []
        opportunity: list[float] = []
        for seed in args.seeds:
            run = measure.measure_run("ordinary", args.grounded_fraction, seed, options)
            coupling = run["coupling"]
            slopes.append(coupling["precision_generation_slope"])
            offspring_corrs.append(coupling["corr_parent_child_offspring"])
            precision_corrs.append(coupling["corr_parent_child_precision"])
            rates.append(run["correct_report_rate"])
            reports_per_adult.append(coupling["mean_reports_per_adult"])
            excess.append(coupling["reproductive_excess"])
            opportunity.append(coupling["opportunity_for_selection"])
        n = len(args.seeds)
        print(
            f"correct_report_attention_value={value:g} "
            f"break_even_precision={args.break_even_precision} "
            f"score_units={args.score_units} "
            f"threshold_range={args.threshold_range} "
            f"correctness_weight={args.correctness_weight:g} "
            f"recruitment_share={args.recruitment_share:g} "
            f"correct-report rate={float(np.mean(rates)):.2%}\n"
            f"  reports/adult lifetime: "
            f"{float(np.mean(reports_per_adult)):.2f}\n"
            f"  reproductive excess: {float(np.mean(excess)):.2f} eligible/slot, "
            f"opportunity for selection I={float(np.mean(opportunity)):.3f}\n"
            f"  clause 1 (within-run rise): mean slope {float(np.mean(slopes)):+.4f}/generation, "
            f"rising in {sum(1 for s in slopes if s > 0)}/{n} seeds\n"
            f"  clause 2 (parent-child offspring r > {_CORRELATION_CLAUSE}): "
            f"mean {float(np.mean(offspring_corrs)):+.3f}, "
            f"cleared in {sum(1 for c in offspring_corrs if c > _CORRELATION_CLAUSE)}/{n} seeds\n"
            f"  heritability of correctness (parent-child precision r): "
            f"mean {float(np.mean(precision_corrs)):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
