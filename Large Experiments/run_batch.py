#!/usr/bin/env python3
"""Batch runner script for TattleTots domain simulations.

This script executes each of the three domain simulations (Coral Key, Fire Ecology,
and Grain Guard) integrated with TattleTots. It reads a batch configuration file,
runs the simulations (sequentially or in parallel), captures logs, and generates
a summary key JSON file mapping each run to its configuration, results, and key metrics.

Usage:
    python run_batch.py --config batch_config.json
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from baseline_parallel import resolve_worker_count, resolve_workspace_root, run_process_pool
from path_safety import KEY_JSON, safe_config_path, safe_output_dir

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = resolve_workspace_root(_SCRIPT_DIR)

# Define paths to the domain repositories relative to the workspace root
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


def run_single_simulation(
    run_name: str,
    domain_key: str,
    config_overrides: Dict[str, Any],
    output_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    """Executes a single simulation run, saving configs, results, and logs."""
    repo_info = REPOS.get(domain_key)
    if not repo_info:
        error_msg = f"Unknown domain: {domain_key}"
        print(f"[-] Error in run '{run_name}': {error_msg}")
        return {
            "status": "failed",
            "error": error_msg,
        }

    print(f"[*] Starting run '{run_name}' ({repo_info['name']})...")

    # 1. Load default config
    workspace_root = _WORKSPACE_ROOT
    default_config_path = workspace_root / repo_info["default_config"]
    if not default_config_path.exists():
        error_msg = f"Default config not found at {default_config_path}"
        print(f"[-] Error in run '{run_name}': {error_msg}")
        return {
            "status": "failed",
            "error": error_msg,
        }

    with open(default_config_path, "r") as f:
        config_data = json.load(f)

    # 2. Apply overrides
    if config_overrides:
        config_data = deep_merge(config_data, config_overrides)

    # 3. Save resolved config in output directory
    resolved_config_path = output_dir / f"{run_name}_config.json"
    with open(resolved_config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # 4. Prepare paths for output and log
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

    # 6. Execute simulation
    start_time = time.time()
    try:
        with open(log_path, "w") as log_file:
            # Write command to log file header
            log_file.write(f"=== Execution Command ===\n{' '.join(cmd)}\n\n")
            log_file.flush()

            # Run subprocess
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(workspace_root),
                check=True,
            )

        elapsed_time = time.time() - start_time
        print(f"[+] Run '{run_name}' completed successfully in {elapsed_time:.1f}s.")

        # 7. Extract key metrics from the output JSON
        metrics = {}
        if results_path.exists():
            try:
                with open(results_path, "r") as r_file:
                    res_data = json.load(r_file)
                
                # Extract standard metrics
                ecology = res_data.get("ecology_metrics", {})
                costs = res_data.get("cost_metrics", {})
                summary = res_data.get("run_summary", {})
                
                metrics = {
                    "steps_completed": summary.get("steps_completed"),
                    "final_population": ecology.get("final_population"),
                    "precision": ecology.get("precision"),
                    "total_cost": costs.get("total_cost"),
                    "mean_cost_per_step": costs.get("mean_cost_per_step"),
                    "domain_specific": res_data.get("domain_metrics", {}),
                }
            except Exception as e:
                print(f"[!] Warning: Failed to parse results file for '{run_name}': {e}")
                metrics = {"error": f"Failed to parse results JSON: {e}"}

        return {
            "status": "success",
            "domain": domain_key,
            "elapsed_seconds": elapsed_time,
            "config_file": resolved_config_path.name,
            "output_file": results_path.name,
            "log_file": log_path.name,
            "metrics": metrics,
        }

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"[-] Run '{run_name}' failed with exit code {e.returncode} after {elapsed_time:.1f}s. Check log: {log_path.name}")
        return {
            "status": "failed",
            "domain": domain_key,
            "elapsed_seconds": elapsed_time,
            "exit_code": e.returncode,
            "config_file": resolved_config_path.name,
            "log_file": log_path.name,
            "error": f"Subprocess failed with exit code {e.returncode}",
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[-] Run '{run_name}' encountered an exception: {e}")
        return {
            "status": "failed",
            "domain": domain_key,
            "elapsed_seconds": elapsed_time,
            "config_file": resolved_config_path.name,
            "error": str(e),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch runner for TattleTots domain simulations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR / "batch_config.json",
        help="Path to batch config JSON file (default: batch_config.json in this directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory path",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel (default: sequential)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(CPU count, job count))",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed simulation logs to stdout (otherwise captured in log files)",
    )
    args = parser.parse_args()

    # 1. Load batch config
    if not args.config.exists():
        print(f"[-] Error: Batch config file not found at {args.config}")
        return 1

    try:
        config_path = safe_config_path(args.config, base=_SCRIPT_DIR)
        with open(config_path, "r") as f:
            batch_config = json.load(f)
    except Exception as e:
        print(f"[-] Error: Failed to parse batch config file: {e}")
        return 1

    # 2. Determine and create output directory
    output_dir_name = args.output_dir or batch_config.get("output_directory", "batch_results")
    output_dir = safe_output_dir(output_dir_name, default_base=_SCRIPT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Results will be saved to: {output_dir}")

    runs = batch_config.get("runs", [])
    if not runs:
        print("[-] Error: No runs specified in batch config.")
        return 1

    n_jobs = len(runs)
    worker_count = resolve_worker_count(args.workers, n_jobs)

    print(f"[*] Found {n_jobs} simulation runs to execute.")
    if args.parallel:
        print(
            f"[*] Execution mode: PARALLEL (ProcessPoolExecutor, "
            f"{worker_count} worker process{'es' if worker_count != 1 else ''}, "
            f"PID {os.getpid()} parent)"
        )
    else:
        print(f"[*] Execution mode: SEQUENTIAL (single process, PID {os.getpid()})")
    print("=" * 60)

    results_key = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "batch_config_file": str(args.config),
        "output_directory": str(output_dir),
        "runs": {},
    }

    start_time = time.time()

    # 3. Execute runs
    submit_kwargs = [
        (
            run["name"],
            run["domain"],
            run.get("config_overrides", {}),
            output_dir,
            args.verbose,
        )
        for run in runs
    ]

    def _store_success(run: Dict[str, Any], res: Dict[str, Any]) -> None:
        results_key["runs"][run["name"]] = res

    def _store_failure(run: Dict[str, Any], exc: Exception) -> None:
        results_key["runs"][run["name"]] = {
            "status": "failed",
            "error": f"Unhandled exception: {exc}",
        }

    if args.parallel:
        run_process_pool(
            run_single_simulation,
            submit_kwargs,
            runs,
            max_workers=worker_count,
            on_success=_store_success,
            on_failure=_store_failure,
        )
    else:
        # Run sequentially
        for run in runs:
            name = run["name"]
            res = run_single_simulation(
                name,
                run["domain"],
                run.get("config_overrides", {}),
                output_dir,
                args.verbose,
            )
            results_key["runs"][name] = res
            print("-" * 40)

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[+] All runs finished in {total_elapsed:.1f}s.")

    # 4. Generate key file
    key_file_path = output_dir / KEY_JSON
    with open(key_file_path, "w") as f:
        json.dump(results_key, f, indent=2)

    print(f"[+] Batch summary key written to: {key_file_path}")
    
    # 5. Print a summary table of the runs
    print("\n=== Batch Execution Summary ===")
    print(f"{'Run Name':<25} | {'Domain':<15} | {'Status':<10} | {'Time (s)':<8} | {'Cost':<10} | {'Population':<10}")
    print("-" * 88)
    for name, run_res in results_key["runs"].items():
        status = run_res.get("status", "unknown")
        domain = run_res.get("domain", "unknown")
        elapsed = f"{run_res.get('elapsed_seconds', 0.0):.1f}"
        
        metrics = run_res.get("metrics", {})
        cost = f"{metrics.get('total_cost', 0.0):.2f}" if "total_cost" in metrics else "N/A"
        pop = str(metrics.get("final_population", "N/A"))
        
        print(f"{name:<25} | {domain:<15} | {status:<10} | {elapsed:<8} | {cost:<10} | {pop:<10}")
    print("=" * 88)

    # Return 0 if all runs succeeded, otherwise 1
    any_failed = any(r.get("status") == "failed" for r in results_key["runs"].values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
