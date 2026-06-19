#!/usr/bin/env python3
"""Run TattleTots-integrated quick validation cases from quick_validation_config.json.

Expands the validation spec (6 scenarios x 3 seeds = 18 BMA runs) into batch
format and executes via run_batch machinery.

Usage:
    python run_quick_validation.py --config path/to/quick_validation_config.json
    python run_quick_validation.py --config ... --probe   # one run per domain (seed 42)
    python run_quick_validation.py --config ... --parallel --workers 3
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent

ADVERSARY_LEVELS: dict[str, dict[str, float]] = {
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


def _sim_overrides(tt_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    sim = copy.deepcopy(tt_cfg)
    sim.pop("seed", None)
    max_steps = sim.pop("max_steps", 800)
    sim["max_steps"] = max_steps
    sim["seed"] = seed
    return sim


def _coral_domain(scenario: dict[str, Any], max_steps: int, seed: int) -> dict[str, Any]:
    adv = ADVERSARY_LEVELS[scenario["adversary_level"]]
    return {
        "total_epochs": max_steps,
        "seed": seed,
        "fleet": {
            "n_iuu_vessels": scenario["iuu_vessels"],
            "underreport_fraction": adv["underreport_fraction"],
        },
        "sensors": {"sar_revisit_interval": scenario["sar_revisit_interval"]},
        "adversary": {
            "ais_disable_probability": adv["ais_disable_probability"],
            "spoof_probability": adv["spoof_probability"],
            "platform_interference_rate": adv["platform_interference_rate"],
        },
    }


def _fire_domain(scenario: dict[str, Any], max_steps: int, seed: int) -> dict[str, Any]:
    return {
        "steps": max_steps,
        "seed": seed,
        "grid_rows": scenario.get("grid_rows", 20),
        "grid_cols": scenario.get("grid_cols", 20),
        "n_cameras": scenario.get("n_cameras", 3),
        "n_drones": scenario.get("n_drones", 0),
        "base_ignition_rate": scenario.get("base_ignition_rate", 0.0001),
        "weather_volatility": scenario.get("weather_volatility", 1.0),
    }


def _grain_domain(
    scenario: dict[str, Any], max_steps: int, seed: int, *, engine_max_dim: int
) -> dict[str, Any]:
    return {
        "steps": max_steps,
        "seed": seed,
        "landscape": scenario["landscape"],
        "grid_rows": scenario.get("grid_rows", 20),
        "grid_cols": scenario.get("grid_cols", 20),
        "pest_intro_probability": scenario.get("pest_intro_probability", 0.02),
        "resistance_initial_frequency": scenario.get("resistance_initial_frequency", 0.01),
        "engine_max_dim": engine_max_dim,
    }


def expand_validation_runs(
    cfg: dict[str, Any],
    *,
    probe: bool = False,
) -> list[dict[str, Any]]:
    """Expand quick_validation_config into run_batch-compatible run entries."""
    tt_cfg = cfg["tattletots_config"]
    seeds = [42] if probe else list(tt_cfg["seed"])
    max_steps = tt_cfg.get("max_steps", 800)
    engine_max_dim = tt_cfg.get("max_stream_dim", 551)

    sections = [
        ("coral_key", "coral_key", "ck", _coral_domain),
        ("fire_ecology", "fire_ecology", "fe", _fire_domain),
    ]
    grain_scenarios = cfg["grain_guard"]["runs"]
    if probe:
        grain_scenarios = grain_scenarios[:1]

    runs: list[dict[str, Any]] = []
    for domain_key, section_key, prefix, domain_fn in sections:
        scenarios = cfg[section_key]["runs"]
        if probe:
            scenarios = scenarios[:1]
        for scenario in scenarios:
            label = scenario["label"]
            for seed in seeds:
                runs.append(
                    {
                        "name": f"{prefix}_{label}_s{seed}",
                        "domain": domain_key,
                        "config_overrides": {
                            "simulation": _sim_overrides(tt_cfg, seed),
                            "domain": domain_fn(scenario, max_steps, seed),
                        },
                    }
                )

    for scenario in grain_scenarios:
        label = scenario["label"]
        for seed in seeds:
            runs.append(
                {
                    "name": f"gg_{label}_s{seed}",
                    "domain": "grain_guard",
                    "config_overrides": {
                        "simulation": _sim_overrides(tt_cfg, seed),
                        "domain": _grain_domain(
                            scenario, max_steps, seed, engine_max_dim=engine_max_dim
                        ),
                    },
                }
            )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick validation runner for TattleTots integration")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to quick_validation_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "quick_validation_results",
        help="Directory for run outputs (default: quick_validation_results/)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run one scenario per domain with seed 42 only (timing probe)",
    )
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--workers", type=int, default=None, help="Worker process count")
    parser.add_argument("--verbose", action="store_true", help="Verbose simulation logs")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"[-] Config not found: {args.config}")
        return 1

    with open(args.config) as f:
        validation_cfg = json.load(f)

    runs = expand_validation_runs(validation_cfg, probe=args.probe)
    batch_cfg = {
        "output_directory": str(args.output_dir),
        "source_config": str(args.config.resolve()),
        "probe_mode": args.probe,
        "runs": runs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = args.output_dir / "quick_validation_batch.json"
    with open(batch_path, "w") as f:
        json.dump(batch_cfg, f, indent=2)

    n_runs = len(runs)
    mode = "probe (3 runs)" if args.probe else f"full validation ({n_runs} runs)"
    print(f"[*] Expanded {mode} -> {batch_path}")

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_batch.py"),
        "--config",
        str(batch_path),
    ]
    if args.parallel:
        cmd.append("--parallel")
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.verbose:
        cmd.append("--verbose")

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
