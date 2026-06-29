"""Command-line interface for running TattleTots simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tattletots.engine.config import GenePoolConfig, SimulationConfig
from tattletots.engine.world import World
from tattletots.path_utils import safe_path_under_base
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario
from tattletots.scenarios.high_dim_shift import HighDimShiftScenario
from tattletots.telemetry.cost_accounting import CostAccumulator

_CWD = Path.cwd()


def _load_scenario(
    name: str, scenario_config: dict[str, int | float | str]
) -> GaussianShiftScenario | HighDimShiftScenario:
    if name == "gaussian_shift":
        return GaussianShiftScenario.from_config(scenario_config)
    if name == "high_dim_shift":
        return HighDimShiftScenario.from_config(scenario_config)
    raise ValueError(f"Unknown scenario: {name}")


def _resolve_config_path(
    raw: Path | None,
) -> tuple[SimulationConfig, dict[str, int | float | str], str, GenePoolConfig | None]:
    gene_pool: GenePoolConfig | None = None
    if raw is not None:
        config_path = safe_path_under_base(raw, _CWD)
        with open(config_path) as f:
            raw_config = json.load(f)
        sim_config = SimulationConfig(**raw_config.get("simulation", {}))
        scenario_config = raw_config.get("scenario", {})
        if "gene_pool" in raw_config:
            gene_pool = GenePoolConfig(**raw_config["gene_pool"])
        scenario_name = str(scenario_config.get("scenario", "gaussian_shift"))
        return sim_config, scenario_config, scenario_name, gene_pool
    return SimulationConfig(), {}, "gaussian_shift", None


def _build_world(
    sim_config: SimulationConfig,
    scenario: GaussianShiftScenario | HighDimShiftScenario,
    gene_pool: GenePoolConfig | None,
) -> World:
    world = World(config=sim_config, gene_pool=gene_pool)
    for stream in scenario.get_streams():
        world.add_stream(stream)
    for user in scenario.get_users():
        world.add_user(user)
    world.seed_population()
    world.set_location_inference(scenario.infer_report_location)
    world.set_dim_to_location(scenario.dim_index_to_location)
    if hasattr(scenario, "get_ground_truth_vector"):
        world.set_ground_truth_vector(scenario.get_ground_truth_vector(0))
    return world


def _run_steps(
    world: World,
    scenario: GaussianShiftScenario | HighDimShiftScenario,
    steps: int,
    *,
    verbose: bool,
    cost_accumulator: CostAccumulator,
) -> None:
    for step_num in range(steps):
        scenario.step(step_num)
        world.set_event_state(scenario.get_active_locations(step_num))
        if hasattr(scenario, "get_ground_truth_vector"):
            world.set_ground_truth_vector(scenario.get_ground_truth_vector(step_num))

        record = world.step()

        if verbose and step_num % 50 == 0:
            print(
                f"  Step {step_num:4d}: pop={record.population:3d} "
                f"births={record.births} deaths={record.deaths} "
                f"reports={record.reports_issued} "
                f"trophic_depth={record.max_trophic_level:.1f} "
                f"working_dim={record.mean_working_dim:.0f}"
            )

        cost_dict = scenario.compute_costs(
            n_escalations=record.reports_issued,
            n_correct=record.correct_reports,
            n_false_alarms=record.false_alarms,
            n_missed=record.missed_events,
        )
        cost_accumulator.record_from_dict(record.time_step, cost_dict)

        if record.population == 0:
            print("  ** Total extinction **")
            break


def _write_results(
    output_path: Path,
    *,
    sim_config: SimulationConfig,
    scenario: GaussianShiftScenario | HighDimShiftScenario,
    world: World,
    cost_summary: dict[str, float],
) -> None:
    summary = world.telemetry.summary()
    output_data = {
        "config": sim_config.model_dump(),
        "scenario": scenario.to_config(),
        "summary": summary,
        "cost_summary": cost_summary,
        "population_history": world.telemetry.population_history(),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Results written to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the tattletots CLI."""
    parser = argparse.ArgumentParser(
        prog="tattletots",
        description="Run a TattleTots dual-currency information ecology simulation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--scenario",
        choices=["gaussian_shift", "high_dim_shift"],
        default="gaussian_shift",
        help="Built-in scenario to run (default: gaussian_shift)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Number of simulation steps (default: 400)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Initial population size (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step progress",
    )

    args = parser.parse_args(argv)

    if args.config:
        sim_config, scenario_config, scenario_name, gene_pool = _resolve_config_path(args.config)
    else:
        sim_config = SimulationConfig(
            initial_population=args.population,
            max_steps=args.steps,
            seed=args.seed,
        )
        scenario_config = {}
        scenario_name = args.scenario
        gene_pool = None

    scenario = _load_scenario(scenario_name, scenario_config)
    world = _build_world(sim_config, scenario, gene_pool)
    cost_accumulator = CostAccumulator()

    print(f"TattleTots v0.1.0 — {scenario_name}")
    print(f"  Population: {sim_config.initial_population}, Steps: {args.steps}, Seed: {args.seed}")
    print()

    _run_steps(world, scenario, args.steps, verbose=args.verbose, cost_accumulator=cost_accumulator)

    summary = world.telemetry.summary()
    cost_summary = cost_accumulator.summary()
    print()
    print("=== Simulation Complete ===")
    print(f"  Final population: {summary['final_population']}")
    print(f"  Peak population:  {summary['peak_population']}")
    print(f"  Total births:     {summary['total_births']}")
    print(f"  Total deaths:     {summary['total_deaths']}")
    print(f"  Reports issued:   {summary['total_reports']}")
    print(f"  Precision:        {summary['precision']:.2%}")
    print(f"  Max trophic depth:{summary['max_trophic_depth']:.1f}")
    print(f"  Equilibrium:      {summary['reached_equilibrium']}")
    print(f"  Total cost:       {cost_summary['total_cost']:.2f}")

    if args.output:
        output_path = safe_path_under_base(args.output, _CWD)
        _write_results(
            output_path,
            sim_config=sim_config,
            scenario=scenario,
            world=world,
            cost_summary=cost_summary,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
