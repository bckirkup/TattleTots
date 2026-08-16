#!/usr/bin/env python3
"""Re-runnable ceiling measurement against any modeled-instrument DomainAdapter.

The adapter is supplied on the command line, so the measurement is not tied to
any one scenario. The default adapter is `SparseSensorScenario`, which publishes
sensor coordinates and moves its latent source, giving a non-vacuous
localization null. Each run emits the `docs/initiation.md` step-2 metrics
(correct-report rate, per-capita attention solvency, grounded-yield share) plus
a `SimulationOutput` payload for cross-domain comparison.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tattletots.engine.config import GenePoolConfig, SimulationConfig
from tattletots.engine.world import World
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.interface.instrument import validate_instrument
from tattletots.interface.reporter_policy import (
    ReporterDecision,
    ReporterPolicyContext,
    register_reporter_policy,
)
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome
from tattletots.models.location import EventLocation
from tattletots.output_schema import EcologyMetrics, RunSummary, SimulationOutput
from tattletots.telemetry.recorder import TelemetryRecorder

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ADAPTER = "tattletots.scenarios.sparse_sensor:SparseSensorScenario"
_ORACLE_POLICY_NAME = "ceiling_oracle_diagnostic_upper_bound"
_INVASION_SHARE = 0.15
_DEFAULT_SEEDS = (42, 43, 44, 45, 46)
_DEFAULT_GROUNDED_FRACTIONS = (0.0, 0.34, 0.67, 1.0)
_DEFAULT_GROUNDED_MULTIPLIERS = (1.0,)
_ALLOWED_OUTPUT_BASES = (_REPO_ROOT, Path(tempfile.gettempdir()).resolve())


def safe_output_path(raw: str) -> Path:
    """Resolve a caller-supplied output path inside an allowed base directory."""
    resolved = (_REPO_ROOT / raw).resolve()
    if not any(resolved.is_relative_to(base) for base in _ALLOWED_OUTPUT_BASES):
        raise ValueError(f"output path escapes the allowed directories: {raw}")
    return resolved


@dataclass
class _OracleDiagnosticPolicy:
    """Harness-local diagnostic upper bound; never a shipped reporter policy."""

    active_locations: tuple[EventLocation, ...] = ()

    def decide(self, _context: ReporterPolicyContext) -> ReporterDecision:
        if not self.active_locations:
            return ReporterDecision(escalate=False)
        return ReporterDecision(escalate=True, location=self.active_locations[0])


register_reporter_policy(_ORACLE_POLICY_NAME, _OracleDiagnosticPolicy)


@dataclass(frozen=True)
class GridPoint:
    """One measured cell: an arm crossed with the grounded-access knobs."""

    arm: str
    grounded_fraction: float
    grounded_multiplier: float

    @property
    def oracle_share(self) -> float:
        if self.arm == "oracle_monoculture":
            return 1.0
        if self.arm == "oracle_invasion":
            return _INVASION_SHARE
        return 0.0

    def label(self) -> str:
        return (
            f"{self.arm}"
            f"|fraction={self.grounded_fraction:g}"
            f"|multiplier={self.grounded_multiplier:g}"
        )


@dataclass
class HarnessOptions:
    """Measurement window and grid supplied by the caller."""

    adapter_spec: str = _DEFAULT_ADAPTER
    steps: int = 200
    seeds: tuple[int, ...] = _DEFAULT_SEEDS
    grounded_fractions: tuple[float, ...] = _DEFAULT_GROUNDED_FRACTIONS
    grounded_multipliers: tuple[float, ...] = _DEFAULT_GROUNDED_MULTIPLIERS
    initial_population: int = 20
    max_population: int = 60
    max_stream_dim: int | None = None
    arms: tuple[str, ...] = ("ordinary", "oracle_monoculture", "oracle_invasion")
    extra_config: dict[str, Any] = field(default_factory=dict)
    gene_pool: dict[str, Any] | None = None


def build_adapter(adapter_spec: str, seed: int, steps: int) -> DomainAdapter:
    """Instantiate `module:Callable`, passing seed/step arguments it accepts."""
    module_name, _, attribute = adapter_spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"adapter must be given as 'module:Callable', got {adapter_spec!r}")
    factory = getattr(importlib.import_module(module_name), attribute)
    parameters = inspect.signature(factory).parameters
    kwargs: dict[str, Any] = {}
    if "seed" in parameters:
        kwargs["seed"] = seed
    for step_name in ("total_steps", "total_epochs", "steps"):
        if step_name in parameters:
            kwargs[step_name] = steps
            break
    adapter = factory(**kwargs)
    if not isinstance(adapter, DomainAdapter):
        raise TypeError(f"{adapter_spec} did not produce a DomainAdapter")
    return adapter


def _simulation_config(point: GridPoint, seed: int, options: HarnessOptions) -> SimulationConfig:
    config_kwargs: dict[str, Any] = {
        "initial_population": options.initial_population,
        "max_population": options.max_population,
        "max_steps": options.steps,
        "seed": seed,
        "grounded_input_fraction": point.grounded_fraction,
        "grounded_attractiveness_multiplier": point.grounded_multiplier,
        **options.extra_config,
    }
    if options.max_stream_dim is not None:
        config_kwargs["max_stream_dim"] = options.max_stream_dim
    return SimulationConfig(**config_kwargs)


def _seed_genomes(world: World, config: SimulationConfig, oracle_share: float) -> list[Genome]:
    genomes = [
        Genome.random_genome(
            world.rng,
            n_streams=max(len(world.streams), 1),
            n_users=max(len(world.users), 1),
            gene_pool=world.gene_pool,
        )
        for _ in range(config.initial_population)
    ]
    n_oracle = int(round(len(genomes) * oracle_share))
    if oracle_share > 0.0:
        n_oracle = max(n_oracle, 1)
    for genome in genomes[:n_oracle]:
        genome.reporter_policy = _ORACLE_POLICY_NAME
    return genomes


def build_world(
    adapter: DomainAdapter,
    point: GridPoint,
    seed: int,
    options: HarnessOptions,
) -> World:
    """Construct a seeded world for one grid cell against the supplied adapter."""
    config = _simulation_config(point, seed, options)
    gene_pool = GenePoolConfig(**options.gene_pool) if options.gene_pool else None
    world = World(config=config, gene_pool=gene_pool)
    for stream in adapter.get_streams():
        world.add_stream(stream)
    for user in adapter.get_users():
        world.add_user(user)
    world.seed_population(genomes=_seed_genomes(world, config, point.oracle_share))
    world.set_location_inference(adapter.infer_report_location)
    world.set_location_frame(adapter.get_location_frame())
    return world


def set_oracle_locations(world: World, active_locations: Sequence[EventLocation]) -> None:
    """Publish the current active locations to any harness-local oracle policy."""
    locations = tuple(active_locations)
    for policy in world.reporter_policies.values():
        if isinstance(policy, _OracleDiagnosticPolicy):
            policy.active_locations = locations


def _correct_report_rate_by_half(telemetry: TelemetryRecorder) -> tuple[float, float]:
    """Correct-report rate over the first and second halves of the run."""
    history = telemetry.history
    if len(history) < 2:
        return 0.0, 0.0
    midpoint = len(history) // 2
    halves = []
    for records in (history[:midpoint], history[midpoint:]):
        reports = sum(record.reports_issued for record in records)
        correct = sum(record.correct_reports for record in records)
        halves.append(correct / reports if reports else 0.0)
    return halves[0], halves[1]


def _offspring_counts(agents: dict[str, Agent]) -> dict[str, int]:
    counts = dict.fromkeys(agents, 0)
    for agent in agents.values():
        for parent_id in agent.state.parent_ids:
            if parent_id in counts:
                counts[parent_id] += 1
    return counts


def parent_child_reproductive_correlation(agents: dict[str, Agent]) -> float:
    """Pearson correlation between a parent's and its child's offspring count."""
    counts = _offspring_counts(agents)
    parent_values: list[float] = []
    child_values: list[float] = []
    for agent in agents.values():
        for parent_id in agent.state.parent_ids:
            if parent_id in counts:
                parent_values.append(float(counts[parent_id]))
                child_values.append(float(counts[agent.id]))
    if len(parent_values) < 3:
        return 0.0
    parents = np.array(parent_values, dtype=np.float64)
    children = np.array(child_values, dtype=np.float64)
    if math.isclose(parents.std(), 0.0, abs_tol=1e-12) or math.isclose(
        children.std(), 0.0, abs_tol=1e-12
    ):
        return 0.0
    return float(np.corrcoef(parents, children)[0, 1])


def _initiation_metrics(world: World) -> dict[str, float]:
    summary = world.telemetry.summary()
    first_half, second_half = _correct_report_rate_by_half(world.telemetry)
    return {
        "correct_report_rate": float(summary["precision"]),
        "correct_report_rate_first_half": first_half,
        "correct_report_rate_second_half": second_half,
        "correct_report_rate_drift": second_half - first_half,
        "per_capita_attention_solvency": float(summary["attention_solvent_fraction"]),
        "grounded_yield_share": float(summary["grounded_yield_share"]),
        "effective_grounded_yield_share": float(summary["effective_grounded_yield_share"]),
        "static_prior_precision": float(summary["static_prior_precision"]),
        "chance_precision": float(summary["chance_precision"]),
        "designed_precision": float(summary["designed_precision"]),
        "ordinary_precision": float(summary["ordinary_precision"]),
        "parent_child_reproductive_correlation": parent_child_reproductive_correlation(
            world.agents
        ),
    }


def _simulation_output(
    world: World,
    adapter: DomainAdapter,
    point: GridPoint,
    seed: int,
) -> SimulationOutput:
    summary = world.telemetry.summary()
    ecology_fields = set(EcologyMetrics.model_fields)
    return SimulationOutput(
        run_summary=RunSummary(
            domain=adapter.__class__.__name__,
            steps_completed=world.telemetry.total_steps,
            seed=seed,
        ),
        simulation_config=world.config.model_dump(),
        domain_config={"arm": point.arm},
        ecology_metrics=EcologyMetrics.model_validate(
            {key: value for key, value in summary.items() if key in ecology_fields}
        ),
    )


def measure_grid_point(
    point: GridPoint,
    seed: int,
    options: HarnessOptions,
) -> tuple[dict[str, Any], SimulationOutput]:
    """Run one arm/knob/seed cell and return its metrics and unified output."""
    adapter = build_adapter(options.adapter_spec, seed, options.steps)
    world = build_world(adapter, point, seed, options)
    for step in range(options.steps):
        adapter.step(step)
        active_locations = adapter.get_active_locations(step)
        world.set_event_state(active_locations)
        set_oracle_locations(world, active_locations)
        world.step()

    metrics: dict[str, Any] = {
        "arm": point.arm,
        "grounded_input_fraction": point.grounded_fraction,
        "grounded_attractiveness_multiplier": point.grounded_multiplier,
        "seed": seed,
        "final_population": int(world.telemetry.summary()["final_population"]),
        "total_reports": int(world.telemetry.total_reports),
        **_initiation_metrics(world),
    }
    return metrics, _simulation_output(world, adapter, point, seed)


def instrument_nulls(options: HarnessOptions) -> dict[str, Any]:
    """Validate the instrument on a fresh adapter before any measurement."""
    adapter = build_adapter(options.adapter_spec, options.seeds[0], options.steps)
    report = validate_instrument(adapter, steps=options.steps)
    return {
        "valid": report.valid,
        "static_prior_baseline": report.static_prior_baseline,
        "chance_baseline": report.chance_baseline,
        "inferability_precision": report.inferability_precision,
        "decoder_precision": report.decoder_precision,
        "candidate_locations": len(report.candidate_locations),
        "distinct_event_locations": report.distinct_event_locations,
        "event_steps": report.event_steps,
        "findings": [
            {
                "check": str(finding.check),
                "passed": finding.passed,
                "message": finding.message,
                "measured": finding.measured,
                "threshold": finding.threshold,
            }
            for finding in report.findings
        ],
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _summarize_cell(runs: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "correct_report_rate",
        "correct_report_rate_drift",
        "per_capita_attention_solvency",
        "grounded_yield_share",
        "effective_grounded_yield_share",
        "designed_precision",
        "ordinary_precision",
        "parent_child_reproductive_correlation",
    )
    summary = {f"mean_{key}": _mean([float(run[key]) for run in runs]) for key in keys}
    summary["mean_final_population"] = _mean([float(run["final_population"]) for run in runs])
    summary["total_reports"] = float(sum(int(run["total_reports"]) for run in runs))
    return summary


def grid_points(options: HarnessOptions) -> list[GridPoint]:
    """Every arm crossed with every grounded-access knob setting."""
    return [
        GridPoint(arm=arm, grounded_fraction=fraction, grounded_multiplier=multiplier)
        for arm in options.arms
        for fraction in options.grounded_fractions
        for multiplier in options.grounded_multipliers
    ]


def run_measurement(options: HarnessOptions) -> dict[str, Any]:
    """Run the full grid and return the measurement payload."""
    nulls = instrument_nulls(options)
    runs: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, list[dict[str, Any]]] = {}
    for point in grid_points(options):
        label = point.label()
        runs[label] = []
        outputs[label] = []
        for seed in options.seeds:
            metrics, output = measure_grid_point(point, seed, options)
            runs[label].append(metrics)
            outputs[label].append(output.model_dump(mode="json"))
    return {
        "adapter": options.adapter_spec,
        "steps": options.steps,
        "seeds": list(options.seeds),
        "instrument": nulls,
        "runs": runs,
        "summary": {label: _summarize_cell(cell) for label, cell in runs.items()},
        "simulation_outputs": outputs,
    }


def falsification_verdict(results: dict[str, Any]) -> dict[str, Any]:
    """Whether any cell clears the initiation falsification test."""
    static_prior = float(results["instrument"]["static_prior_baseline"])
    cleared: list[dict[str, Any]] = []
    for label, cell in results["summary"].items():
        rising = cell["mean_correct_report_rate_drift"] > 0.0 and (
            cell["mean_correct_report_rate"] > static_prior
        )
        heritable = cell["mean_parent_child_reproductive_correlation"] > 0.2
        if rising or heritable:
            cleared.append(
                {
                    "cell": label,
                    "correct_report_rate_rose": rising,
                    "reproductive_correlation_above_threshold": heritable,
                    "mean_correct_report_rate": cell["mean_correct_report_rate"],
                    "mean_parent_child_reproductive_correlation": (
                        cell["mean_parent_child_reproductive_correlation"]
                    ),
                }
            )
    return {
        "static_prior_null": static_prior,
        "localization_non_vacuous": bool(results["instrument"]["valid"]),
        "cells_clearing_falsification_test": cleared,
        "passed": bool(cleared),
    }


def markdown_report(results: dict[str, Any], verdict: dict[str, Any]) -> str:
    """Render the measured grid as a markdown table."""
    lines = [
        "# Ceiling measurement on a modeled instrument",
        "",
        f"- Adapter: `{results['adapter']}`",
        f"- Steps per run: `{results['steps']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in results['seeds'])}`",
        f"- Static-prior null: **{results['instrument']['static_prior_baseline']:.2%}**",
        f"- Uniform (chance) null: **{results['instrument']['chance_baseline']:.2%}**",
        f"- Evidence inferability: **{results['instrument']['inferability_precision']:.2%}**",
        f"- Instrument valid: **{results['instrument']['valid']}**",
        "",
        "| Cell | Correct-report rate | Drift (2nd half − 1st) | Attention solvency | "
        "Grounded-yield share | Parent–child repro corr | Reports |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, cell in results["summary"].items():
        lines.append(
            f"| `{label}` | {cell['mean_correct_report_rate']:.2%} | "
            f"{cell['mean_correct_report_rate_drift']:+.2%} | "
            f"{cell['mean_per_capita_attention_solvency']:.2%} | "
            f"{cell['mean_grounded_yield_share']:.2%} | "
            f"{cell['mean_parent_child_reproductive_correlation']:.3f} | "
            f"{int(cell['total_reports'])} |"
        )
    lines.extend(
        [
            "",
            "## Falsification verdict",
            "",
            f"- Cleared: **{verdict['passed']}**",
            f"- Cells clearing the test: `{len(verdict['cells_clearing_falsification_test'])}`",
            "",
            "The oracle arms are harness-local diagnostic upper bounds, not shipped "
            "reporter policies. A cell clears the test when the correct-report rate "
            "rises across the run above the static-prior null without changing initial "
            "parameters, or when parent–child reproductive correlation exceeds 0.2.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=_DEFAULT_ADAPTER, help="module:Callable")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(_DEFAULT_SEEDS))
    parser.add_argument(
        "--grounded-fractions",
        type=float,
        nargs="+",
        default=list(_DEFAULT_GROUNDED_FRACTIONS),
    )
    parser.add_argument(
        "--grounded-multipliers",
        type=float,
        nargs="+",
        default=list(_DEFAULT_GROUNDED_MULTIPLIERS),
    )
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    parser.add_argument("--max-stream-dim", type=int, default=None)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["ordinary", "oracle_monoculture", "oracle_invasion"],
    )
    parser.add_argument("--output", default="docs/ceiling-measurement.json")
    parser.add_argument("--report", default="docs/ceiling-measurement.md")
    return parser.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> HarnessOptions:
    """Translate parsed arguments into harness options."""
    return HarnessOptions(
        adapter_spec=args.adapter,
        steps=args.steps,
        seeds=tuple(args.seeds),
        grounded_fractions=tuple(args.grounded_fractions),
        grounded_multipliers=tuple(args.grounded_multipliers),
        initial_population=args.initial_population,
        max_population=args.max_population,
        max_stream_dim=args.max_stream_dim,
        arms=tuple(args.arms),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write the JSON payload and markdown report."""
    args = _parse_args(argv)
    options = options_from_args(args)
    results = run_measurement(options)
    verdict = falsification_verdict(results)
    results["falsification"] = verdict

    output_path = safe_output_path(args.output)
    report_path = safe_output_path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(results, verdict))
    print(markdown_report(results, verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
