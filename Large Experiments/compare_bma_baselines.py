#!/usr/bin/env python3
"""Compare BMA quick-validation results against archived A0-A3 baselines.

Usage:
    python compare_bma_baselines.py
    python compare_bma_baselines.py --bma-key path/to/quick_validation_results/key.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline_spot_check import resolve_key_json

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE = _SCRIPT_DIR.parent.parent

# Quick-validation scenario -> baseline run name template (seed substituted)
SCENARIO_BASELINE_PATTERNS: dict[str, dict[str, str]] = {
    "coral_key": {
        "medium_adversary_3iuu": "ck_baselines_iuu3_advmedium_sar8_s{seed}",
        "high_adversary_3iuu": "ck_baselines_iuu3_advhigh_sar8_s{seed}",
    },
    "fire_ecology": {
        "medium_fire_10drones": "fe_baselines_phase_0_drop0_d10_ignmedium_wxmedium_s{seed}",
        "high_fire_15drones": "fe_baselines_phase_0_drop0_d15_ignhigh_wxmedium_s{seed}",
    },
    "grain_guard": {
        "monoculture_baseline": "gg_baselines_monoculture_pestmedium_weedmedium_res0p01_s{seed}",
        "orchard_complexity": "gg_baselines_orchard_pestmedium_weedmedium_res0p01_s{seed}",
    },
}

# BMA run name prefix -> scenario label
BMA_RUN_PREFIXES: dict[str, tuple[str, str]] = {
    "ck_medium_adversary_3iuu": ("coral_key", "medium_adversary_3iuu"),
    "ck_high_adversary_3iuu": ("coral_key", "high_adversary_3iuu"),
    "fe_medium_fire_10drones": ("fire_ecology", "medium_fire_10drones"),
    "fe_high_fire_15drones": ("fire_ecology", "high_fire_15drones"),
    "gg_monoculture_baseline": ("grain_guard", "monoculture_baseline"),
    "gg_orchard_complexity": ("grain_guard", "orchard_complexity"),
}


def _load_baseline_runs(domain: str) -> dict[str, Any]:
    key_path = resolve_key_json(domain)
    return json.loads(key_path.read_text()).get("runs", {})


def _parse_bma_run(name: str) -> tuple[str, str, int] | None:
    if not name.endswith(tuple(f"_s{s}" for s in range(1000, 1100)) + tuple(f"_s{s}" for s in (42, 43, 44))):
        pass
    if "_s" not in name:
        return None
    base, seed_str = name.rsplit("_s", 1)
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    for prefix, (domain, scenario) in BMA_RUN_PREFIXES.items():
        if base == prefix or name.startswith(prefix):
            return domain, scenario, seed
    return None


def _arch_short_name(arch: str) -> str:
    """Map full architecture label to short id (A0, A1, ...)."""
    if arch.startswith("A0"):
        return "A0"
    if arch.startswith("A1"):
        return "A1"
    if arch.startswith("A2"):
        return "A2"
    if arch.startswith("A3"):
        return "A3"
    return arch


def _extract_bma_metric(domain: str, metrics: dict[str, Any]) -> float | None:
    dom = metrics.get("domain_specific") or metrics.get("domain_metrics") or metrics
    if domain == "coral_key":
        return dom.get("iuu_detection_rate") or dom.get("detection_rate")
    if domain == "fire_ecology":
        return dom.get("total_burned_area") or dom.get("burned_area")
    if domain == "grain_guard":
        return dom.get("final_yield") or dom.get("mean_yield")
    return None


def _baseline_best_and_targets(
    domain: str,
    baseline_summary: dict[str, dict[str, float]],
    qv_cfg: dict[str, Any],
    scenario: str,
) -> dict[str, Any]:
    section = {
        "coral_key": "coral_key",
        "fire_ecology": "fire_ecology",
        "grain_guard": "grain_guard",
    }[domain]
    targets: dict[str, float] = {}
    for run in qv_cfg.get(section, {}).get("runs", []):
        if run["label"] == scenario:
            targets = run.get("baseline_target", {})
            break

    if domain == "coral_key":
        metric_key = "detection_rate"
        bma_higher_better = True
    elif domain == "fire_ecology":
        metric_key = "burned_cells"
        bma_higher_better = False
    else:
        metric_key = "final_yield"
        bma_higher_better = True

    arch_values = {
        arch: float(vals.get(metric_key, 0.0)) for arch, vals in baseline_summary.items()
    }
    if domain == "fire_ecology":
        best_baseline = min(arch_values.items(), key=lambda x: x[1])
    else:
        best_baseline = max(arch_values.items(), key=lambda x: x[1])

    return {
        "metric": metric_key,
        "arch_values": arch_values,
        "best_baseline_arch": best_baseline[0],
        "best_baseline_value": best_baseline[1],
        "baseline_targets": targets,
        "bma_higher_better": bma_higher_better,
    }


def _bma_wins(
    bma_val: float,
    *,
    best_baseline: float,
    targets: dict[str, float],
    arch_values: dict[str, float],
    higher_better: bool,
) -> tuple[bool, bool]:
    """Return (beats_best_baseline, beats_all_configured_targets)."""
    if higher_better:
        beats_best = bma_val > best_baseline
        if targets:
            beats_targets = all(
                bma_val > float(target_val)
                for short_id, target_val in targets.items()
                if any(_arch_short_name(a) == short_id for a in arch_values)
            )
        else:
            beats_targets = beats_best
    else:
        beats_best = bma_val < best_baseline
        if targets:
            beats_targets = all(
                bma_val < float(target_val)
                for short_id, target_val in targets.items()
                if any(_arch_short_name(a) == short_id for a in arch_values)
            )
        else:
            beats_targets = beats_best
    return beats_best, beats_targets


def compare(
    bma_key_path: Path,
    qv_config_path: Path,
) -> dict[str, Any]:
    bma_key = json.loads(bma_key_path.read_text())
    qv_cfg = json.loads(qv_config_path.read_text())

    baseline_cache: dict[str, dict[str, Any]] = {
        d: _load_baseline_runs(d) for d in ("coral_key", "fire_ecology", "grain_guard")
    }

    comparisons: list[dict[str, Any]] = []
    for run_name, run_entry in bma_key.get("runs", {}).items():
        if run_entry.get("status") != "success":
            comparisons.append({"run_name": run_name, "status": "failed", "error": run_entry.get("error")})
            continue

        parsed = _parse_bma_run(run_name)
        if parsed is None:
            continue
        domain, scenario, seed = parsed
        pattern = SCENARIO_BASELINE_PATTERNS[domain][scenario]
        baseline_name = pattern.format(seed=seed)
        baseline_run = baseline_cache[domain].get(baseline_name)
        if baseline_run is None:
            comparisons.append(
                {
                    "run_name": run_name,
                    "status": "missing_baseline",
                    "baseline_name": baseline_name,
                }
            )
            continue

        bma_metrics = run_entry.get("metrics", {})
        bma_val = _extract_bma_metric(domain, bma_metrics)
        if bma_val is None:
            # Try loading full results file
            out_file = bma_key_path.parent / run_entry.get("output_file", "")
            if out_file.is_file():
                full = json.loads(out_file.read_text())
                bma_val = _extract_bma_metric(domain, {"domain_specific": full.get("domain_metrics", {})})

        baseline_info = _baseline_best_and_targets(
            domain,
            baseline_run["baselines_summary"],
            qv_cfg,
            scenario,
        )
        beats_best, beats_targets = _bma_wins(
            float(bma_val) if bma_val is not None else float("nan"),
            best_baseline=baseline_info["best_baseline_value"],
            targets=baseline_info["baseline_targets"],
            arch_values=baseline_info["arch_values"],
            higher_better=baseline_info["bma_higher_better"],
        )

        comparisons.append(
            {
                "run_name": run_name,
                "domain": domain,
                "scenario": scenario,
                "seed": seed,
                "baseline_run": baseline_name,
                "bma_metric": baseline_info["metric"],
                "bma_value": bma_val,
                "baseline_arch_values": baseline_info["arch_values"],
                "best_baseline_arch": baseline_info["best_baseline_arch"],
                "best_baseline_value": baseline_info["best_baseline_value"],
                "baseline_targets": baseline_info["baseline_targets"],
                "beats_best_baseline": beats_best,
                "beats_all_targets": beats_targets,
                "status": "success",
            }
        )

    n_ok = sum(1 for c in comparisons if c.get("status") == "success")
    n_win_best = sum(1 for c in comparisons if c.get("beats_best_baseline"))
    n_win_targets = sum(1 for c in comparisons if c.get("beats_all_targets"))

    return {
        "bma_key": str(bma_key_path),
        "qv_config": str(qv_config_path),
        "total_comparisons": len(comparisons),
        "successful": n_ok,
        "beats_best_baseline_count": n_win_best,
        "beats_all_targets_count": n_win_targets,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare BMA quick-validation vs archived baselines")
    parser.add_argument(
        "--bma-key",
        type=Path,
        default=_SCRIPT_DIR / "quick_validation_results" / "key.json",
    )
    parser.add_argument(
        "--qv-config",
        type=Path,
        default=_WORKSPACE / "quick_validation_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "quick_validation_results" / "bma_baseline_comparison.json",
    )
    args = parser.parse_args()

    if not args.bma_key.exists():
        print(f"[-] BMA key not found: {args.bma_key}")
        return 1

    report = compare(args.bma_key, args.qv_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[+] Compared {report['successful']} BMA runs")
    print(f"    Beats best baseline: {report['beats_best_baseline_count']}/{report['successful']}")
    print(f"    Beats all targets:   {report['beats_all_targets_count']}/{report['successful']}")
    print(f"[+] Report: {args.output}")

    print("\n=== Per-run summary ===")
    for c in report["comparisons"]:
        if c.get("status") != "success":
            print(f"  {c.get('run_name')}: {c.get('status')}")
            continue
        win = "WIN" if c["beats_best_baseline"] else "LOSE"
        print(
            f"  [{win}] {c['run_name']}: BMA {c['bma_metric']}={c['bma_value']} "
            f"vs best {c['best_baseline_arch']}={c['best_baseline_value']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
