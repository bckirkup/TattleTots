#!/usr/bin/env python3
"""Statistical spot-check for archived A0-A3 baseline scans.

Samples factor-settings from archived key.json files, re-runs each in triplicate
with new seeds, and checks equivalence-by-tolerance (1% relative) against the
archived triplicate reference means.

Usage:
    python baseline_spot_check.py --domain all --sample 100 --workers 8
    python baseline_spot_check.py --domain grain_guard --sample 10 --workers 4
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from baseline_parallel import resolve_worker_count, resolve_workspace_root, run_process_pool
from path_safety import KEY_JSON, safe_output_dir

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = resolve_workspace_root(_SCRIPT_DIR)

EPS0 = 1e-9
REL_TOL = 0.01
STEPS = 800
EPOCHS = 800
GRID_ROWS = 20
GRID_COLS = 20

# Quick-validation factor-settings (must be included in sample)
MANDATORY_METADATA: dict[str, list[dict[str, Any]]] = {
    "fire_ecology": [
        {
            "deployment_phase": "phase_0",
            "sensor_dropout": "0%",
            "drone_fleet_size": 10,
            "ignition_rate": "medium",
            "weather_volatility": "medium",
            "n_cameras": 3,
        },
        {
            "deployment_phase": "phase_0",
            "sensor_dropout": "0%",
            "drone_fleet_size": 15,
            "ignition_rate": "high",
            "weather_volatility": "medium",
            "n_cameras": 3,
        },
    ],
    "grain_guard": [
        {
            "landscape": "monoculture",
            "pest_pressure": "medium",
            "weed_pressure": "medium",
            "resistance_initial_frequency": 0.01,
        },
        {
            "landscape": "orchard",
            "pest_pressure": "medium",
            "weed_pressure": "medium",
            "resistance_initial_frequency": 0.01,
        },
    ],
    "coral_key": [
        {
            "iuu_vessel_count": 3,
            "adversary_level": "medium",
            "sar_revisit_interval": 8,
        },
        {
            "iuu_vessel_count": 3,
            "adversary_level": "high",
            "sar_revisit_interval": 8,
        },
    ],
}


def _metadata_key(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)


def _load_runner_module(relative_script: str, module_name: str) -> Any:
    path = _WORKSPACE_ROOT / relative_script
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runner: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_key_json(domain: str) -> Path:
    """Find archived key.json (handles nested output directories)."""
    candidates = {
        "fire_ecology": [
            _WORKSPACE_ROOT / "fire_ecology_baselines_results" / KEY_JSON,
        ],
        "grain_guard": [
            _WORKSPACE_ROOT / "grain_guard_baselines_results" / KEY_JSON,
            _WORKSPACE_ROOT
            / "grain_guard_baselines_results"
            / "grain_guard_baselines_results"
            / KEY_JSON,
        ],
        "coral_key": [
            _WORKSPACE_ROOT / "coral_key_baselines_results" / KEY_JSON,
            _WORKSPACE_ROOT
            / "coral_key_baselines_results"
            / "coral_key_baselines_results"
            / KEY_JSON,
        ],
    }
    for path in candidates[domain]:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No {KEY_JSON} found for domain {domain}")


def load_domain_configs() -> dict[str, dict[str, Any]]:
    return {
        "fire_ecology": json.loads(
            (_WORKSPACE_ROOT / "Scrapiron_and_the_Bear/baselines/fire_ecology_baselines_config.json").read_text()
        ),
        "grain_guard": json.loads(
            (_WORKSPACE_ROOT / "Xylella_SPQR/baselines/grain_guard_baselines_config.json").read_text()
        ),
        "coral_key": json.loads(
            (
                _WORKSPACE_ROOT
                / "Coral_Key_in_Three_Hour_Epochs/baselines/coral_key_baselines_config.json"
            ).read_text()
        ),
    }


def group_runs_by_factor(key_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group archived runs by metadata (factor-setting), keyed by metadata JSON."""
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"metadata": {}, "by_seed": {}})
    for run_name, run_entry in key_data.get("runs", {}).items():
        if run_entry.get("status") != "success":
            continue
        metadata = run_entry.get("metadata", {})
        fkey = _metadata_key(metadata)
        groups[fkey]["metadata"] = metadata
        # Extract seed from run name suffix _s{seed}
        if "_s" in run_name:
            seed_str = run_name.rsplit("_s", 1)[-1]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            groups[fkey]["by_seed"][seed] = run_entry.get("baselines_summary", {})
    return dict(groups)


def sample_factor_settings(
    groups: dict[str, dict[str, Any]],
    *,
    domain: str,
    sample_n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Return factor-setting keys to test, always including mandatory QV settings."""
    all_keys = list(groups.keys())
    mandatory_keys: list[str] = []
    for meta in MANDATORY_METADATA.get(domain, []):
        mk = _metadata_key(meta)
        if mk in groups:
            mandatory_keys.append(mk)

    remaining = [k for k in all_keys if k not in mandatory_keys]
    n_extra = max(0, min(sample_n, len(all_keys)) - len(mandatory_keys))
    sampled_extra: list[str] = []
    if remaining and n_extra > 0:
        n_pick = min(n_extra, len(remaining))
        pick_idx = rng.choice(len(remaining), size=n_pick, replace=False)
        sampled_extra = [remaining[int(i)] for i in np.atleast_1d(pick_idx)]
    selected = list(dict.fromkeys(mandatory_keys + sampled_extra))
    return selected[: min(sample_n, len(all_keys))]


def _fire_run_kwargs(metadata: dict[str, Any], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    ignition_map = cfg.get("ignition_rates", {"medium": 0.0001})
    volatility_map = cfg.get("weather_volatility", {"medium": 1.0})
    return {
        "steps": cfg.get("steps", STEPS),
        "seed": seed,
        "n_cameras": int(metadata.get("n_cameras", 3)),
        "n_drones": int(metadata["drone_fleet_size"]),
        "grid_rows": cfg.get("grid_rows", GRID_ROWS),
        "grid_cols": cfg.get("grid_cols", GRID_COLS),
        "base_ignition_rate": float(ignition_map[metadata["ignition_rate"]]),
        "weather_volatility": float(volatility_map[metadata["weather_volatility"]]),
    }


def _grain_run_kwargs(metadata: dict[str, Any], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    pest_map = cfg["pest_pressure_levels"]
    weed_map = cfg["weed_pressure_levels"]
    pest = pest_map[metadata["pest_pressure"]]
    weed = weed_map[metadata["weed_pressure"]]
    return {
        "steps": cfg.get("steps", STEPS),
        "seed": seed,
        "landscape": metadata["landscape"],
        "grid_rows": cfg.get("grid_rows", GRID_ROWS),
        "grid_cols": cfg.get("grid_cols", GRID_COLS),
        "pest_intro_probability": float(pest["intro_probability"]),
        "pest_density_boost": float(pest["density_boost"]),
        "weed_density_base": float(weed["density_base"]),
        "resistance_initial_frequency": float(metadata["resistance_initial_frequency"]),
    }


def _coral_run_kwargs(metadata: dict[str, Any], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    adv = cfg["adversary_levels"][metadata["adversary_level"]]
    return {
        "run_name": f"spot_ck_s{seed}",
        "epochs": cfg.get("epochs", EPOCHS),
        "seed": seed,
        "iuu_vessels": int(metadata["iuu_vessel_count"]),
        "sar_revisit": int(metadata["sar_revisit_interval"]),
        "adv_params": adv,
    }


def _summarize_fire(res: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "detections": float(d["detections"]),
            "suppressions": float(d["suppressions"]),
            "burned_cells": float(d["burned_cells"]),
            "cost": float(d["cost"]),
        }
        for name, d in res["baselines"].items()
    }


def _summarize_grain(res: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "final_yield": float(d["final_yield"]),
            "spray_volume_L": float(d["spray_volume_L"]),
            "false_sprays": float(d["false_sprays"]),
            "total_cost": float(d["total_cost"]),
        }
        for name, d in res["baselines"].items()
    }


def _summarize_coral(res: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "detection_rate": float(d["detection_rate"]),
            "false_alarm_rate": float(d["false_alarm_rate"]),
            "patrol_cost": float(d["patrol_cost"]),
        }
        for name, d in res["baselines"].items()
    }


DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "fire_ecology": {
        "label": "Fire Ecology",
        "run_kwargs_fn": _fire_run_kwargs,
        "summarize_fn": _summarize_fire,
        "runner_script": "Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py",
        "runner_module": "fe_baselines_runner",
    },
    "grain_guard": {
        "label": "Grain Guard",
        "run_kwargs_fn": _grain_run_kwargs,
        "summarize_fn": _summarize_grain,
        "runner_script": "Xylella_SPQR/baselines/run_grain_guard_baselines.py",
        "runner_module": "gg_baselines_runner",
    },
    "coral_key": {
        "label": "Coral Key",
        "run_kwargs_fn": _coral_run_kwargs,
        "summarize_fn": _summarize_coral,
        "runner_script": "Coral_Key_in_Three_Hour_Epochs/baselines/run_coral_key_baselines.py",
        "runner_module": "ck_baselines_runner",
    },
}


def _mean_summary(
    summaries: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}
    archs = summaries[0].keys()
    out: dict[str, dict[str, float]] = {}
    for arch in archs:
        metrics = summaries[0][arch].keys()
        out[arch] = {
            m: float(np.mean([s[arch][m] for s in summaries if arch in s])) for m in metrics
        }
    return out


def check_equivalence_pair(
    ref_val: float,
    test_val: float,
    *,
    rel_tol: float = REL_TOL,
    eps0: float = EPS0,
) -> tuple[bool, float]:
    denom = max(abs(ref_val), eps0)
    rel_err = abs(test_val - ref_val) / denom
    return rel_err <= rel_tol, rel_err


def check_equivalence(
    ref_mean: dict[str, dict[str, float]],
    test_mean: dict[str, dict[str, float]],
    *,
    rel_tol: float = REL_TOL,
    eps0: float = EPS0,
) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for arch, ref_metrics in ref_mean.items():
        test_metrics = test_mean.get(arch, {})
        for metric, ref_val in ref_metrics.items():
            test_val = test_metrics.get(metric)
            if test_val is None:
                failures.append(
                    {
                        "architecture": arch,
                        "metric": metric,
                        "reason": "missing_in_test",
                        "ref_mean": ref_val,
                        "test_mean": None,
                    }
                )
                continue
            ok, rel_err = check_equivalence_pair(ref_val, test_val, rel_tol=rel_tol, eps0=eps0)
            if not ok:
                failures.append(
                    {
                        "architecture": arch,
                        "metric": metric,
                        "ref_mean": ref_val,
                        "test_mean": test_val,
                        "rel_err": rel_err,
                        "rel_tol": rel_tol,
                    }
                )
    return len(failures) == 0, failures


def check_summary_against_archived(
    archived: dict[str, dict[str, float]],
    rerun: dict[str, dict[str, float]],
    *,
    rel_tol: float = REL_TOL,
    eps0: float = EPS0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare one rerun summary to one archived baselines_summary."""
    failures: list[dict[str, Any]] = []
    for arch, ref_metrics in archived.items():
        test_metrics = rerun.get(arch, {})
        for metric, ref_val in ref_metrics.items():
            test_val = test_metrics.get(metric)
            if test_val is None:
                failures.append(
                    {
                        "architecture": arch,
                        "metric": metric,
                        "reason": "missing_in_rerun",
                        "archived": ref_val,
                    }
                )
                continue
            ok, rel_err = check_equivalence_pair(float(ref_val), float(test_val), rel_tol=rel_tol, eps0=eps0)
            if not ok:
                failures.append(
                    {
                        "architecture": arch,
                        "metric": metric,
                        "archived": ref_val,
                        "rerun": test_val,
                        "rel_err": rel_err,
                    }
                )
    return len(failures) == 0, failures


def check_envelope_equivalence(
    ref_values_by_metric: dict[str, dict[str, list[float]]],
    test_mean: dict[str, dict[str, float]],
    *,
    rel_tol: float = REL_TOL,
    eps0: float = EPS0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Pass if test_mean falls within archived triplicate min/max expanded by rel_tol."""
    failures: list[dict[str, Any]] = []
    for arch, metrics in ref_values_by_metric.items():
        for metric, ref_vals in metrics.items():
            test_val = test_mean.get(arch, {}).get(metric)
            if test_val is None:
                failures.append({"architecture": arch, "metric": metric, "reason": "missing_in_test"})
                continue
            lo = min(ref_vals)
            hi = max(ref_vals)
            margin = rel_tol * max(abs(lo), abs(hi), eps0)
            if not (lo - margin <= test_val <= hi + margin):
                failures.append(
                    {
                        "architecture": arch,
                        "metric": metric,
                        "ref_min": lo,
                        "ref_max": hi,
                        "test_mean": test_val,
                        "margin": margin,
                    }
                )
    return len(failures) == 0, failures


def _make_worker(domain: str) -> Callable[..., dict[str, Any]]:
    spec = DOMAIN_SPECS[domain]
    mod = _load_runner_module(spec["runner_script"], spec["runner_module"])
    return mod.run_single_simulation


def _execute_spot_job(domain: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    factor_key = kwargs.pop("_factor_key")
    worker = _make_worker(domain)
    res = worker(**kwargs)
    summarize = DOMAIN_SPECS[domain]["summarize_fn"]
    return {
        "domain": domain,
        "seed": kwargs["seed"],
        "factor_key": factor_key,
        "summary": summarize(res),
    }


def run_domain_spot_check(
    domain: str,
    *,
    sample_n: int,
    ref_seeds: list[int],
    test_seed_offset: int,
    rng: np.random.Generator,
    workers: int | None,
    domain_cfgs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spec = DOMAIN_SPECS[domain]
    key_path = resolve_key_json(domain)
    key_data = json.loads(key_path.read_text())
    groups = group_runs_by_factor(key_data)
    selected_keys = sample_factor_settings(groups, domain=domain, sample_n=sample_n, rng=rng)

    test_seeds = [s + test_seed_offset for s in ref_seeds]
    cfg = domain_cfgs[domain]
    kwargs_fn = spec["run_kwargs_fn"]

    jobs: list[dict[str, Any]] = []
    job_meta: list[dict[str, Any]] = []
    for fkey in selected_keys:
        group = groups[fkey]
        metadata = group["metadata"]
        for seed in ref_seeds + test_seeds:
            kw = kwargs_fn(metadata, seed, cfg)
            kw["_factor_key"] = fkey
            jobs.append(kw)
            tag = "ref" if seed in ref_seeds else "test"
            job_meta.append(
                {
                    "name": f"{domain}_{hash(fkey) & 0xFFFF:04x}_{tag}_s{seed}",
                    "factor_key": fkey,
                    "seed": seed,
                    "seed_group": tag,
                    "metadata": metadata,
                }
            )

    worker_count = resolve_worker_count(workers, len(jobs))
    print(f"[*] {spec['label']}: {len(selected_keys)} factor-settings, {len(jobs)} reruns")
    print(f"    Reference: {key_path}")
    print(f"    Ref seeds (determinism): {ref_seeds}")
    print(f"    Test seeds (distribution): {test_seeds}")

    results_by_factor_seed: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(dict)
    failures_exec: list[str] = []

    def _on_success(_run: dict[str, Any], res: dict[str, Any]) -> None:
        results_by_factor_seed[res["factor_key"]][res["seed"]] = res["summary"]

    def _on_failure(_run: dict[str, Any], exc: Exception) -> None:
        failures_exec.append(str(exc))

    t0 = time.time()
    submit_args = [(domain, kw) for kw in jobs]
    run_process_pool(
        _execute_spot_job,
        submit_args,
        job_meta,
        max_workers=worker_count,
        on_success=_on_success,
        on_failure=_on_failure,
    )
    elapsed = time.time() - t0

    factor_results: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    for fkey in selected_keys:
        group = groups[fkey]
        ref_by_seed = group["by_seed"]

        determinism_failures: list[dict[str, Any]] = []
        for seed in ref_seeds:
            archived = ref_by_seed.get(seed)
            rerun = results_by_factor_seed[fkey].get(seed)
            if archived is None or rerun is None:
                determinism_failures.append(
                    {"seed": seed, "reason": "missing_archived_or_rerun"}
                )
                continue
            ok, fails = check_summary_against_archived(archived, rerun)
            if not ok:
                determinism_failures.extend([{"seed": seed, **f} for f in fails[:3]])

        ref_summaries = [ref_by_seed[s] for s in ref_seeds if s in ref_by_seed]
        ref_mean = _mean_summary(ref_summaries)
        test_summaries = [
            results_by_factor_seed[fkey][s] for s in test_seeds if s in results_by_factor_seed[fkey]
        ]
        test_mean = _mean_summary(test_summaries) if test_summaries else {}

        mean_ok, mean_failures = check_equivalence(ref_mean, test_mean) if test_mean else (False, [])
        ref_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for summary in ref_summaries:
            for arch, metrics in summary.items():
                for metric, val in metrics.items():
                    ref_values[arch][metric].append(float(val))
        env_ok, env_failures = (
            check_envelope_equivalence(ref_values, test_mean) if test_mean else (False, [])
        )

        passed = not determinism_failures and len(ref_summaries) == len(ref_seeds)
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        factor_results.append(
            {
                "factor_key": fkey,
                "metadata": group["metadata"],
                "passed": passed,
                "determinism_failures": determinism_failures[:10],
                "new_seed_mean_passed": mean_ok,
                "new_seed_mean_failures": mean_failures[:5],
                "new_seed_envelope_passed": env_ok,
                "new_seed_envelope_failures": env_failures[:5],
                "worst_det_rel_err": max(
                    (f.get("rel_err", 0.0) for f in determinism_failures), default=0.0
                ),
                "worst_mean_rel_err": max(
                    (f.get("rel_err", 0.0) for f in mean_failures), default=0.0
                ),
            }
        )

    domain_passed = n_fail == 0 and not failures_exec
    return {
        "domain": domain,
        "label": spec["label"],
        "reference_key": str(key_path),
        "sampled_factor_settings": len(selected_keys),
        "test_runs": len(jobs),
        "passed_factor_settings": n_pass,
        "failed_factor_settings": n_fail,
        "execution_failures": failures_exec,
        "elapsed_seconds": elapsed,
        "passed": domain_passed,
        "factor_results": factor_results,
        "failed_factors": [f for f in factor_results if not f.get("passed")][:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline statistical spot-check (equivalence-by-tolerance)")
    parser.add_argument(
        "--domain",
        choices=[*DOMAIN_SPECS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--sample", type=int, default=100, help="Factor-settings to sample per domain")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for factor-setting sampling")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ref-seeds", type=str, default="42,43,44")
    parser.add_argument("--test-seed-offset", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "baseline_spot_check_results",
    )
    args = parser.parse_args(argv)

    ref_seeds = [int(s.strip()) for s in args.ref_seeds.split(",")]
    rng = np.random.default_rng(args.seed)
    domain_cfgs = load_domain_configs()

    domains = list(DOMAIN_SPECS.keys()) if args.domain == "all" else [args.domain]
    output_dir = safe_output_dir(args.output_dir, default_base=_WORKSPACE_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "rel_tolerance": REL_TOL,
        "eps0": EPS0,
        "gate": "determinism_same_seed_1pct",
        "note": (
            "Pass requires archived vs same-seed rerun equivalence (1% relative). "
            "New-seed mean/envelope checks are reported but non-gating."
        ),
        "ref_seeds": ref_seeds,
        "test_seeds": [s + args.test_seed_offset for s in ref_seeds],
        "sample_per_domain": args.sample,
        "sampling_seed": args.seed,
        "domains": {},
        "all_passed": True,
    }

    exit_code = 0
    for domain in domains:
        print("=" * 60)
        try:
            result = run_domain_spot_check(
                domain,
                sample_n=args.sample,
                ref_seeds=ref_seeds,
                test_seed_offset=args.test_seed_offset,
                rng=rng,
                workers=args.workers,
                domain_cfgs=domain_cfgs,
            )
        except Exception as exc:
            print(f"[-] {domain} spot-check failed: {exc}")
            result = {"domain": domain, "passed": False, "error": str(exc)}
            exit_code = 1
            report["all_passed"] = False
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{status}] {result['label']}: "
                f"{result['passed_factor_settings']}/{result['sampled_factor_settings']} "
                f"factor-settings equivalent ({result['elapsed_seconds']:.1f}s)"
            )
            if not result["passed"]:
                exit_code = 1
                report["all_passed"] = False
                for ff in result.get("failed_factors", [])[:5]:
                    print(
                        f"    FAIL metadata={ff.get('metadata')} "
                        f"worst_det={ff.get('worst_det_rel_err', 'n/a')}"
                    )

        report["domains"][domain] = result

    report_path = output_dir / "spot_check_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print("=" * 60)
    print(f"[+] Report written to {report_path}")
    if report["all_passed"]:
        print("[+] All domains passed equivalence spot-check.")
    else:
        print("[-] One or more domains failed spot-check.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
