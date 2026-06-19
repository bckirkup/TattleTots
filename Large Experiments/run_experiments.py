#!/usr/bin/env python3
"""Designed Experiments Runner for TattleTots domain simulations.

This script sets up and executes the designed experiments described in
`Designed Experiments.txt` and `Design of Experiment.md`. It supports:
1. Sweeping the 5 TattleTots shared parameter levels (Conservative to Exploratory).
2. Sweeping domain-specific factors mapped to actual codebase parameters.
3. Running in triplicate (3 seeds per configuration).
4. Running for 800 steps/epochs per run.
5. Setting `max_stream_dim` to 1000.
6. Parallel execution using ThreadPoolExecutor.
7. A fast `--smoke-test` mode to verify the configuration and execution pipeline.

Usage:
    # Run a fast smoke test to verify everything works:
    python run_experiments.py --smoke-test

    # Run the full suite (warning: will take a long time due to combinatorial sweep):
    python run_experiments.py --parallel
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Define the 5 TattleTots shared parameter levels interpolated between Conservative and Exploratory
TATTLETOTS_LEVELS = [
    # Level 1 (Conservative)
    {
        "initial_population": 50,
        "max_population": 100,
        "max_stream_dim": 1000,  # Hardcoded to 1000 as per specification
        "initial_info_energy": 1.5,
        "initial_attn_energy": 1.5,
        "false_alarm_penalty": 0.8,
        "trust_delta_neg": 0.4,
        "trust_delta_pos": 0.05,
        "mutation_rate": 0.05,
        "recombination_probability": 0.2,
    },
    # Level 2 (Interpolated)
    {
        "initial_population": 100,
        "max_population": 200,
        "max_stream_dim": 1000,
        "initial_info_energy": 1.625,
        "initial_attn_energy": 1.625,
        "false_alarm_penalty": 0.675,
        "trust_delta_neg": 0.35,
        "trust_delta_pos": 0.05,
        "mutation_rate": 0.075,
        "recombination_probability": 0.25,
    },
    # Level 3 (Balanced)
    {
        "initial_population": 150,
        "max_population": 300,
        "max_stream_dim": 1000,
        "initial_info_energy": 1.75,
        "initial_attn_energy": 1.75,
        "false_alarm_penalty": 0.55,
        "trust_delta_neg": 0.3,
        "trust_delta_pos": 0.05,
        "mutation_rate": 0.10,
        "recombination_probability": 0.3,
    },
    # Level 4 (Interpolated)
    {
        "initial_population": 200,
        "max_population": 400,
        "max_stream_dim": 1000,
        "initial_info_energy": 1.875,
        "initial_attn_energy": 1.875,
        "false_alarm_penalty": 0.425,
        "trust_delta_neg": 0.25,
        "trust_delta_pos": 0.05,
        "mutation_rate": 0.125,
        "recombination_probability": 0.35,
    },
    # Level 5 (Exploratory)
    {
        "initial_population": 250,
        "max_population": 500,
        "max_stream_dim": 1000,
        "initial_info_energy": 2.0,
        "initial_attn_energy": 2.0,
        "false_alarm_penalty": 0.3,
        "trust_delta_neg": 0.2,
        "trust_delta_pos": 0.05,
        "mutation_rate": 0.15,
        "recombination_probability": 0.4,
    },
]

# Map domain names to their script and default config paths
REPOS = {
    "coral_key": {
        "name": "Coral Key (ReefWatch)",
        "script": "Coral_Key_in_Three_Hour_Epochs/scripts/run_with_tattletots.py",
        "default_config": "Coral_Key_in_Three_Hour_Epochs/configs/tattletots_integration.json",
    },
    "fire_ecology": {
        "name": "Fire Ecology",
        "script": "Scrapiron_and_the_Bear/scripts/run_with_tattletots.py",
        "default_config": "Scrapiron_and_the_Bear/configs/tattletots_integration.json",
    },
    "grain_guard": {
        "name": "Grain Guard",
        "script": "Xylella_SPQR/scripts/run_with_tattletots.py",
        "default_config": "Xylella_SPQR/configs/tattletots_integration.json",
    },
}


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dict2 into dict1 in place."""
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def generate_experiment_configs(smoke_test: bool = False) -> Dict[str, Any]:
    """Generates the full designed experiments configuration structure."""
    config = {
        "output_directory": "designed_experiments_results" if not smoke_test else "smoke_test_results",
        "steps": 5 if smoke_test else 800,
        "seeds": [42] if smoke_test else [42, 43, 44],  # Triplicate runs
        "tattletots_levels": TATTLETOTS_LEVELS if not smoke_test else [TATTLETOTS_LEVELS[2]],  # Level 3 for smoke test
        "domains": {
            "coral_key": {
                "factors": {
                    "iuu_vessel_count": [1, 3, 6] if not smoke_test else [3],
                    "adversary_level": ["low", "medium", "high"] if not smoke_test else ["medium"],
                    "sar_revisit_interval": [4, 8, 16] if not smoke_test else [8],
                    "stream_dimension": [551, 1000] if not smoke_test else [1000],
                },
                "adversary_levels": {
                    "low": {
                        "ais_disable_probability": 0.3,
                        "spoof_probability": 0.1,
                        "underreport_fraction": 0.05,
                        "platform_interference_rate": 0.00,
                    },
                    "medium": {
                        "ais_disable_probability": 0.7,
                        "spoof_probability": 0.3,
                        "underreport_fraction": 0.15,
                        "platform_interference_rate": 0.05,
                    },
                    "high": {
                        "ais_disable_probability": 0.9,
                        "spoof_probability": 0.6,
                        "underreport_fraction": 0.30,
                        "platform_interference_rate": 0.15,
                    },
                }
            },
            "fire_ecology": {
                "factors": {
                    "deployment_phase": ["phase_0", "phase_1", "phase_2", "phase_3"] if not smoke_test else ["phase_2"],
                    "sensor_dropout": ["0%", "20%", "50%"] if not smoke_test else ["0%"],
                    "stream_dimension": [400, 1000, 5000] if not smoke_test else [1000],
                },
                "phases": {
                    "phase_0": {"n_cameras": 3, "n_weather_stations": 4, "n_fuel_sensors": 0},
                    "phase_1": {"n_cameras": 8, "n_weather_stations": 4, "n_fuel_sensors": 0},
                    "phase_2": {"n_cameras": 12, "n_weather_stations": 6, "n_fuel_sensors": 4},
                    "phase_3": {"n_cameras": 12, "n_weather_stations": 8, "n_fuel_sensors": 6},
                }
            },
            "grain_guard": {
                "factors": {
                    "landscape": ["monoculture", "orchard", "intercrop"] if not smoke_test else ["monoculture"],
                    "sensor_budget": ["sparse", "medium", "dense"] if not smoke_test else ["medium"],
                    "stream_dimension": [117, 500, 1000] if not smoke_test else [1000],
                },
                "sensor_budgets": {
                    "sparse": {"n_traps": 5, "n_weather_stations": 1, "n_soil_sensors": 2},
                    "medium": {"n_traps": 10, "n_weather_stations": 2, "n_soil_sensors": 4},
                    "dense": {"n_traps": 20, "n_weather_stations": 4, "n_soil_sensors": 8},
                }
            }
        }
    }
    return config


def run_single_run(
    run_name: str,
    domain_key: str,
    sim_config: Dict[str, Any],
    domain_config: Dict[str, Any],
    output_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    """Executes a single run of a designed experiment."""
    repo_info = REPOS[domain_key]
    workspace_root = Path(__file__).parent.resolve()
    
    # 1. Load default config
    default_config_path = workspace_root / repo_info["default_config"]
    with open(default_config_path, "r") as f:
        config_data = json.load(f)

    # 2. Merge simulation and domain configs
    config_data["simulation"] = deep_merge(config_data.get("simulation", {}), sim_config)
    config_data["domain"] = deep_merge(config_data.get("domain", {}), domain_config)

    # 3. Save resolved config
    resolved_config_path = output_dir / f"{run_name}_config.json"
    with open(resolved_config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # 4. Prepare paths
    results_path = output_dir / f"{run_name}_results.json"
    log_path = output_dir / f"{run_name}.log"
    script_path = workspace_root / repo_info["script"]

    # 5. Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(resolved_config_path),
        "--output",
        str(results_path),
    ]
    if verbose:
        cmd.append("--verbose")

    # 6. Execute subprocess
    start_time = time.time()
    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"=== Execution Command ===\n{' '.join(cmd)}\n\n")
            log_file.flush()

            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(workspace_root),
                check=True,
            )

        elapsed_time = time.time() - start_time
        
        # Extract key metrics
        metrics = {}
        if results_path.exists():
            with open(results_path, "r") as r_file:
                res_data = json.load(r_file)
            metrics = {
                "steps_completed": res_data.get("run_summary", {}).get("steps_completed"),
                "final_population": res_data.get("ecology_metrics", {}).get("final_population"),
                "precision": res_data.get("ecology_metrics", {}).get("precision"),
                "total_cost": res_data.get("cost_metrics", {}).get("total_cost"),
                "domain_specific": res_data.get("domain_metrics", {}),
            }

        return {
            "status": "success",
            "domain": domain_key,
            "elapsed_seconds": elapsed_time,
            "config_file": resolved_config_path.name,
            "output_file": results_path.name,
            "log_file": log_path.name,
            "metrics": metrics,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "status": "failed",
            "domain": domain_key,
            "elapsed_seconds": elapsed_time,
            "config_file": resolved_config_path.name,
            "log_file": log_path.name,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Designed Experiments Runner for TattleTots"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to save/load designed experiments config JSON",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast smoke test of the designed experiments suite",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel",
    )
    parser.add_argument(
        "--domain",
        choices=["coral_key", "fire_ecology", "grain_guard"],
        help="Limit execution to a single domain",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs",
    )
    args = parser.parse_args()

    # 1. Generate or load config
    config_path = args.config or Path("designed_experiments_config.json")
    if not config_path.exists() or args.smoke_test:
        exp_config = generate_experiment_configs(args.smoke_test)
        if not args.smoke_test:
            with open(config_path, "w") as f:
                json.dump(exp_config, f, indent=2)
            print(f"[+] Generated designed experiments configuration at: {config_path}")
    else:
        with open(config_path, "r") as f:
            exp_config = json.load(f)
        print(f"[+] Loaded designed experiments configuration from: {config_path}")

    # 2. Setup output directory
    output_dir = Path(exp_config["output_directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Results will be saved to: {output_dir}")

    # 3. Build the list of runs to execute
    runs_to_execute = []
    steps = exp_config["steps"]
    seeds = exp_config["seeds"]
    tattletots_levels = exp_config["tattletots_levels"]
    domains_data = exp_config["domains"]

    for domain_key, domain_data in domains_data.items():
        if args.domain and domain_key != args.domain:
            continue

        factors = domain_data["factors"]

        if domain_key == "coral_key":
            # Generate combinatorial runs for Coral Key
            for iuu in factors["iuu_vessel_count"]:
                for adv in factors["adversary_level"]:
                    for sar in factors["sar_revisit_interval"]:
                        for stream_dim in factors["stream_dimension"]:
                            for level_idx, tt_level in enumerate(tattletots_levels):
                                for seed in seeds:
                                    run_name = f"ck_iuu{iuu}_adv{adv}_sar{sar}_dim{stream_dim}_ttL{level_idx+1}_s{seed}"
                                    
                                    # Build configs
                                    sim_cfg = tt_level.copy()
                                    sim_cfg["max_steps"] = steps
                                    sim_cfg["seed"] = seed

                                    adv_params = domain_data["adversary_levels"][adv]
                                    dom_cfg = {
                                        "total_epochs": steps,
                                        "seed": seed,
                                        "fleet": {
                                            "n_iuu_vessels": iuu,
                                            "underreport_fraction": adv_params["underreport_fraction"],
                                        },
                                        "sensors": {
                                            "sar_revisit_interval": sar,
                                        },
                                        "adversary": {
                                            "ais_disable_probability": adv_params["ais_disable_probability"],
                                            "spoof_probability": adv_params["spoof_probability"],
                                            "platform_interference_rate": adv_params["platform_interference_rate"],
                                        }
                                    }
                                    runs_to_execute.append({
                                        "name": run_name,
                                        "domain": domain_key,
                                        "sim_config": sim_cfg,
                                        "domain_config": dom_cfg,
                                    })

        elif domain_key == "fire_ecology":
            # Generate combinatorial runs for Fire Ecology
            for phase in factors["deployment_phase"]:
                for dropout in factors["sensor_dropout"]:
                    for stream_dim in factors["stream_dimension"]:
                        for level_idx, tt_level in enumerate(tattletots_levels):
                            for seed in seeds:
                                run_name = f"fe_ph{phase}_drop{dropout.replace('%','')}_dim{stream_dim}_ttL{level_idx+1}_s{seed}"
                                
                                # Build configs
                                sim_cfg = tt_level.copy()
                                sim_cfg["max_steps"] = steps
                                sim_cfg["seed"] = seed

                                phase_params = domain_data["phases"][phase].copy()
                                # Apply dropout reduction to sensors
                                dropout_frac = float(dropout.replace("%", "")) / 100.0
                                phase_params["n_cameras"] = max(0, int(phase_params["n_cameras"] * (1.0 - dropout_frac)))
                                phase_params["n_weather_stations"] = max(0, int(phase_params["n_weather_stations"] * (1.0 - dropout_frac)))
                                phase_params["n_fuel_sensors"] = max(0, int(phase_params["n_fuel_sensors"] * (1.0 - dropout_frac)))

                                dom_cfg = {
                                    "steps": steps,
                                    "seed": seed,
                                    "n_cameras": phase_params["n_cameras"],
                                    "n_weather_stations": phase_params["n_weather_stations"],
                                    "n_fuel_sensors": phase_params["n_fuel_sensors"],
                                    "max_thermal_dim": stream_dim,
                                }
                                runs_to_execute.append({
                                    "name": run_name,
                                    "domain": domain_key,
                                    "sim_config": sim_cfg,
                                    "domain_config": dom_cfg,
                                })

        elif domain_key == "grain_guard":
            # Generate combinatorial runs for Grain Guard
            for landscape in factors["landscape"]:
                for budget in factors["sensor_budget"]:
                    for stream_dim in factors["stream_dimension"]:
                        for level_idx, tt_level in enumerate(tattletots_levels):
                            for seed in seeds:
                                run_name = f"gg_ls{landscape}_bud{budget}_dim{stream_dim}_ttL{level_idx+1}_s{seed}"
                                
                                # Build configs
                                sim_cfg = tt_level.copy()
                                sim_cfg["max_steps"] = steps
                                sim_cfg["seed"] = seed

                                budget_params = domain_data["sensor_budgets"][budget]
                                dom_cfg = {
                                    "steps": steps,
                                    "seed": seed,
                                    "landscape": landscape,
                                    "n_traps": budget_params["n_traps"],
                                    "n_weather_stations": budget_params["n_weather_stations"],
                                    "n_soil_sensors": budget_params["n_soil_sensors"],
                                    "engine_max_dim": stream_dim,
                                }
                                runs_to_execute.append({
                                    "name": run_name,
                                    "domain": domain_key,
                                    "sim_config": sim_cfg,
                                    "domain_config": dom_cfg,
                                })

    print(f"[*] Generated {len(runs_to_execute)} total run configurations.")
    print(f"[*] Execution mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}")
    print("=" * 60)

    results_key = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_smoke_test": args.smoke_test,
        "output_directory": str(output_dir),
        "runs": {},
    }

    start_time = time.time()

    # 4. Execute runs
    if args.parallel:
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    run_single_run,
                    run["name"],
                    run["domain"],
                    run["sim_config"],
                    run["domain_config"],
                    output_dir,
                    args.verbose,
                ): run["name"]
                for run in runs_to_execute
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    results_key["runs"][name] = res
                    if args.verbose:
                        print(f"[+] Completed: {name}")
                except Exception as e:
                    print(f"[-] Run '{name}' raised an unhandled exception: {e}")
                    results_key["runs"][name] = {
                        "status": "failed",
                        "error": f"Unhandled exception: {e}",
                    }
    else:
        for run in runs_to_execute:
            name = run["name"]
            res = run_single_run(
                name,
                run["domain"],
                run["sim_config"],
                run["domain_config"],
                output_dir,
                args.verbose,
            )
            results_key["runs"][name] = res
            print(f"[+] Finished: {name} (Status: {res['status']}, Time: {res.get('elapsed_seconds', 0.0):.1f}s)")

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[+] All runs finished in {total_elapsed:.1f}s.")

    # 5. Save the summary key file
    key_file_path = output_dir / "key.json"
    with open(key_file_path, "w") as f:
        json.dump(results_key, f, indent=2)

    print(f"[+] Designed experiments summary key written to: {key_file_path}")

    # Print summary table
    print("\n=== Designed Experiments Execution Summary ===")
    print(f"{'Run Name':<45} | {'Domain':<15} | {'Status':<10} | {'Time (s)':<8} | {'Population':<10}")
    print("-" * 98)
    for name, run_res in results_key["runs"].items():
        status = run_res.get("status", "unknown")
        domain = run_res.get("domain", "unknown")
        elapsed = f"{run_res.get('elapsed_seconds', 0.0):.1f}"
        metrics = run_res.get("metrics", {})
        pop = str(metrics.get("final_population", "N/A"))
        print(f"{name:<45} | {domain:<15} | {status:<10} | {elapsed:<8} | {pop:<10}")
    print("=" * 98)

    any_failed = any(r.get("status") == "failed" for r in results_key["runs"].values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
