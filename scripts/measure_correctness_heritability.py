#!/usr/bin/env python3
"""Is report correctness heritable at all, and if not, why not?

`docs/payoff-fix-measurement.md` closed the two payoff breaks and exposed a third:
paying for correctness raises correctness -> offspring but not parent-child
offspring correlation, because among evolved agents the parent-child correlation of
precision is ~0.02-0.13 while the heritable-by-construction oracle lineage measures
~0.6-0.7. Selection has almost nothing to act on. Two candidate causes:

  (a) estimation noise -- with a low decoder ceiling and a sparse event base rate,
      an individual agent issues too few verified reports for its own precision to
      be a usable estimate of its genome's precision, so lineage differences are
      swamped by sampling luck;
  (b) no genomic leverage -- precision is simply not a function of the genome, so
      there is no lineage difference to estimate in the first place.

Three measurements separate them, all on the supplied adapter:

  1. Report-count conditioning: parent-child precision correlation restricted to
     agents with at least K verified reports. Rising with K implicates (a).
  2. Excess variance: observed between-agent variance of precision against the
     binomial variance expected from each agent's report count. A ratio near 1.0
     means the spread is pure sampling noise.
  3. Clone repeatability: replicate monoculture runs of single seed genomes. The
     intraclass correlation of generation-0 precision across genomes versus across
     replicate seeds is genomic leverage measured directly, free of both selection
     and estimation-noise confounds. An ICC near 0 implicates (b).

Prints only; it writes no artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from tattletots.models.genome import Genome
from tattletots.telemetry.payoff_ledger import PayoffLedger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "scripts" / "run_ceiling_measurement.py"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ceiling_measurement", _HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load harness from {_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _run_world(
    arm: str,
    seed: int,
    options: Any,
    grounded_fraction: float,
    genomes: list[Genome] | None = None,
) -> tuple[Any, PayoffLedger]:  # noqa: PLR0913 - one call site per measurement
    """Run one world, optionally overriding the seeded genomes, with a ledger."""
    point = harness.GridPoint(arm=arm, grounded_fraction=grounded_fraction, grounded_multiplier=1.0)
    adapter = harness.build_adapter(options.adapter_spec, seed, options.steps)
    world = harness.build_world(adapter, point, seed, options)
    if genomes is not None:
        world.agents.clear()
        world.seed_population(genomes=[genome.model_copy(deep=True) for genome in genomes])
    ledger = PayoffLedger()
    for step in range(options.steps):
        adapter.step(step)
        active = adapter.get_active_locations(step)
        world.set_event_state(active)
        harness.set_oracle_locations(world, active)
        world.step()
        ledger.observe(world)
    ledger.finalize(world)
    return world, ledger


def _precision(reports: int, correct: int) -> float:
    return correct / reports if reports else 0.0


def report_count_conditioning(
    ledgers: Sequence[PayoffLedger],
    thresholds: Sequence[int],
) -> dict[int, tuple[float, int]]:
    """Parent-child precision correlation among agents with >= K reports."""
    conditioned: dict[int, tuple[float, int]] = {}
    for threshold in thresholds:
        parents: list[float] = []
        children: list[float] = []
        for ledger in ledgers:
            records = {record.agent_id: record for record in ledger.records}
            for record in records.values():
                if record.reports_issued < threshold:
                    continue
                for parent_id in record.parent_ids:
                    parent = records.get(parent_id)
                    if parent is None or parent.reports_issued < threshold:
                        continue
                    parents.append(_precision(parent.reports_issued, parent.correct_reports))
                    children.append(_precision(record.reports_issued, record.correct_reports))
        conditioned[threshold] = (_pearson(parents, children), len(parents))
    return conditioned


def excess_variance(ledgers: Sequence[PayoffLedger], min_reports: int = 5) -> dict[str, float]:
    """Observed between-agent precision variance against its binomial noise floor."""
    precisions: list[float] = []
    counts: list[float] = []
    for ledger in ledgers:
        for record in ledger.records:
            if record.reports_issued < min_reports:
                continue
            precisions.append(_precision(record.reports_issued, record.correct_reports))
            counts.append(float(record.reports_issued))
    if len(precisions) < 3:
        return {"n_agents": float(len(precisions))}
    observed = float(np.var(precisions, ddof=1))
    pooled = float(np.mean(precisions))
    # Noise floor: if every agent shared the pooled precision, its own estimate would
    # still scatter binomially around it with its own report count.
    expected = float(np.mean([pooled * (1.0 - pooled) / count for count in counts]))
    return {
        "n_agents": float(len(precisions)),
        "mean_precision": float(np.mean(precisions)),
        "observed_variance": observed,
        "binomial_noise_variance": expected,
        "excess_variance_ratio": observed / expected if expected > 0 else float("nan"),
    }


def reporting_opportunity(ledgers: Sequence[PayoffLedger]) -> dict[str, float]:
    """How many verified reports a single agent's precision estimate rests on."""
    counts = [
        float(record.reports_issued)
        for ledger in ledgers
        for record in ledger.records
        if record.adult_steps > 0
    ]
    adult_steps = [
        float(record.adult_steps)
        for ledger in ledgers
        for record in ledger.records
        if record.adult_steps > 0
    ]
    if not counts:
        return {"n_adults": 0.0}
    array = np.asarray(counts)
    return {
        "n_adults": float(array.size),
        "mean_reports_per_adult": float(array.mean()),
        "median_reports_per_adult": float(np.median(array)),
        "max_reports_per_adult": float(array.max()),
        "share_with_at_least_5_reports": float((array >= 5).mean()),
        "mean_adult_steps": float(np.mean(adult_steps)),
    }


def _run_precision(world: Any) -> tuple[float, float]:
    """Whole-run correct-report rate and report count over all agents ever alive."""
    reports = sum(agent.state.reports_issued for agent in world.agents.values())
    correct = sum(agent.state.correct_reports for agent in world.agents.values())
    return _precision(reports, correct), float(reports)


def clone_repeatability(
    options: Any,
    grounded_fraction: float,
    n_genomes: int,
    replicates: Sequence[int],
) -> dict[str, float]:
    """Between-genome versus within-genome (replicate) variance of founder precision."""
    clone_options = replace(
        options,
        extra_config={
            **options.extra_config,
            "mutation_rate": 0.0,
            "recombination_probability": 0.0,
        },
    )
    rng = np.random.default_rng(0)
    scaffold_world = harness.build_world(
        harness.build_adapter(options.adapter_spec, replicates[0], options.steps),
        harness.GridPoint(
            arm="ordinary", grounded_fraction=grounded_fraction, grounded_multiplier=1.0
        ),
        replicates[0],
        clone_options,
    )
    seed_genomes = [
        Genome.random_genome(
            rng,
            n_streams=max(len(scaffold_world.streams), 1),
            n_users=max(len(scaffold_world.users), 1),
            gene_pool=scaffold_world.gene_pool,
        )
        for _ in range(n_genomes)
    ]

    rows: list[list[float]] = []
    report_counts: list[float] = []
    for genome in seed_genomes:
        clones = [genome for _ in range(clone_options.initial_population)]
        row: list[float] = []
        for seed in replicates:
            world, _ = _run_world(
                "ordinary", seed, clone_options, grounded_fraction, genomes=clones
            )
            precision, reports = _run_precision(world)
            row.append(precision)
            report_counts.append(reports)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float64)
    group_means = matrix.mean(axis=1)
    between = float(np.var(group_means, ddof=1)) if len(group_means) > 1 else 0.0
    within = float(np.mean(matrix.var(axis=1, ddof=1))) if matrix.shape[1] > 1 else 0.0
    total = between + within
    return {
        "n_genomes": float(matrix.shape[0]),
        "n_replicates": float(matrix.shape[1]),
        "mean_clone_run_precision": float(matrix.mean()),
        "between_genome_variance": between,
        "within_genome_variance": within,
        "intraclass_correlation": between / total if total > 0 else float("nan"),
        "min_genome_mean": float(group_means.min()),
        "max_genome_mean": float(group_means.max()),
        "mean_reports_per_run": float(np.mean(report_counts)),
    }


def attenuation_prediction(
    genomic_variance: float,
    pooled_precision: float,
    reports_per_agent: float,
    clone_icc: float,
) -> dict[str, float]:
    """Parent-child precision correlation predicted from the sample size alone.

    A per-agent precision estimate built on n reports carries binomial error variance
    p(1-p)/n on top of the genomic variance, so a genuinely heritable trait measures
    attenuated by var_g / (var_g + p(1-p)/n). Comparing the prediction with the
    observed evolved correlation says whether the weak signal needs any explanation
    beyond how few reports one agent issues.
    """
    if genomic_variance <= 0.0 or reports_per_agent <= 0.0:
        return {}
    noise = pooled_precision * (1.0 - pooled_precision) / reports_per_agent
    attenuation = genomic_variance / (genomic_variance + noise)
    reports_for_half = (
        pooled_precision * (1.0 - pooled_precision) / genomic_variance
        if genomic_variance > 0
        else float("nan")
    )
    return {
        "per_report_noise_variance": noise,
        "attenuation_factor": attenuation,
        "predicted_parent_child_precision_r": clone_icc * attenuation,
        "reports_needed_to_halve_attenuation": reports_for_half,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", default="tattletots.scenarios.sparse_sensor:SparseSensorScenario"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--grounded-fraction", type=float, default=0.67)
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    parser.add_argument("--report-thresholds", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    parser.add_argument("--clone-genomes", type=int, default=6)
    parser.add_argument("--clone-replicates", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--arm",
        default="ordinary",
        help="Arm for the evolved-population measurements (1) and (2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the three break-3 measurements and print them."""
    args = _parse_args(argv)
    options = harness.HarnessOptions(
        adapter_spec=args.adapter,
        steps=args.steps,
        seeds=tuple(args.seeds),
        initial_population=args.initial_population,
        max_population=args.max_population,
    )

    ledgers = [
        _run_world(args.arm, seed, options, args.grounded_fraction)[1] for seed in args.seeds
    ]

    print(f"arm={args.arm} steps={args.steps} seeds={list(args.seeds)}")
    print("1. parent-child precision correlation, conditioned on report count")
    for threshold, (correlation, pairs) in report_count_conditioning(
        ledgers, args.report_thresholds
    ).items():
        print(f"   >= {threshold:>3} reports: r={correlation:+.3f} over {pairs} pairs")

    print("2. reporting opportunity per agent, and the binomial noise floor")
    opportunity = reporting_opportunity(ledgers)
    spread = excess_variance(ledgers, min_reports=1)
    for key, value in {**opportunity, **spread}.items():
        print(f"   {key}: {value:.4f}")

    print("3. clone repeatability of run precision (genomic leverage)")
    clones = clone_repeatability(
        options, args.grounded_fraction, args.clone_genomes, args.clone_replicates
    )
    for key, value in clones.items():
        print(f"   {key}: {value:.4f}")

    print("4. parent-child precision correlation predicted by sample size alone")
    for key, value in attenuation_prediction(
        clones.get("between_genome_variance", 0.0),
        spread.get("mean_precision", 0.0),
        opportunity.get("mean_reports_per_adult", 0.0),
        clones.get("intraclass_correlation", 0.0),
    ).items():
        print(f"   {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
