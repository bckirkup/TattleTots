"""Command-line interface for running TattleTots simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tattletots.engine.config import GenePoolConfig, SimulationConfig
from tattletots.engine.world import World
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario
from tattletots.scenarios.high_dim_shift import HighDimShiftScenario
from tattletots.telemetry.cost_accounting import CostAccumulator


def _load_scenario(
    name: str, scenario_config: dict[str, int | float | str]
) -> GaussianShiftScenario | HighDimShiftScenario:
    if name == "gaussian_shift":
        return GaussianShiftScenario.from_config(scenario_config)
    if name == "high_dim_shift":
        return HighDimShiftScenario.from_config(scenario_config)
    raise ValueError(f"Unknown scenario: {name}")


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

    gene_pool: GenePoolConfig | None = None
    if args.config:
        with open(args.config) as f:
            raw_config = json.load(f)
        sim_config = SimulationConfig(**raw_config.get("simulation", {}))
        scenario_config = raw_config.get("scenario", {})
        if "gene_pool" in raw_config:
            gene_pool = GenePoolConfig(**raw_config["gene_pool"])
        scenario_name = scenario_config.get("scenario", args.scenario)
    else:
        sim_config = SimulationConfig(
            initial_population=args.population,
            max_steps=args.steps,
            seed=args.seed,
        )
        scenario_config = {}
        scenario_name = args.scenario

    scenario = _load_scenario(str(scenario_name), scenario_config)

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

    cost_accumulator = CostAccumulator()

    print(f"TattleTots v0.1.0 — {scenario_name}")
    print(f"  Population: {sim_config.initial_population}, Steps: {args.steps}, Seed: {args.seed}")
    print()

    for step_num in range(args.steps):
        scenario.step(step_num)
        world.set_event_state(scenario.get_active_locations(step_num))
        if hasattr(scenario, "get_ground_truth_vector"):
            world.set_ground_truth_vector(scenario.get_ground_truth_vector(step_num))

        record = world.step()

        if args.verbose and step_num % 50 == 0:
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
        output_data = {
            "config": sim_config.model_dump(),
            "scenario": scenario.to_config(),
            "summary": summary,
            "cost_summary": cost_summary,
            "population_history": world.telemetry.population_history(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
