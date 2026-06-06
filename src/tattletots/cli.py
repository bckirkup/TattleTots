"""Command-line interface for running TattleTots simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


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
        choices=["gaussian_shift"],
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

    # Load or build configuration
    if args.config:
        with open(args.config) as f:
            raw_config = json.load(f)
        sim_config = SimulationConfig(**raw_config.get("simulation", {}))
        scenario_config = raw_config.get("scenario", {})
    else:
        sim_config = SimulationConfig(
            initial_population=args.population,
            max_steps=args.steps,
            seed=args.seed,
        )
        scenario_config = {}

    # Build scenario
    if args.scenario == "gaussian_shift":
        scenario = GaussianShiftScenario.from_config(scenario_config)
    else:
        print(f"Unknown scenario: {args.scenario}", file=sys.stderr)
        return 1

    # Initialize world
    world = World(config=sim_config)
    for stream in scenario.get_streams():
        world.add_stream(stream)
    for user in scenario.get_users():
        world.add_user(user)
    world.seed_population()

    # Run simulation
    print(f"TattleTots v0.1.0 — {args.scenario}")
    print(f"  Population: {sim_config.initial_population}, Steps: {args.steps}, Seed: {args.seed}")
    print()

    for step_num in range(args.steps):
        # Advance domain
        scenario.step(step_num)
        world.set_ground_truth(scenario.get_ground_truth(step_num))

        # Advance ecology
        record = world.step()

        if args.verbose and step_num % 50 == 0:
            print(
                f"  Step {step_num:4d}: pop={record.population:3d} "
                f"births={record.births} deaths={record.deaths} "
                f"reports={record.reports_issued} "
                f"trophic_depth={record.max_trophic_level:.1f}"
            )

        if record.population == 0:
            print("  ** Total extinction **")
            break

    # Summary
    summary = world.telemetry.summary()
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

    # Write output
    if args.output:
        output_data = {
            "config": sim_config.model_dump(),
            "scenario": scenario.to_config(),
            "summary": summary,
            "population_history": world.telemetry.population_history(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
