#!/usr/bin/env python3
"""Run baseline parameter scans (A0–A3) across all three domains.

Each domain has its own runner under ``{domain}/baselines/``. This script
invokes them in sequence (or one domain at a time) from the workspace root.

Usage:
    python run_all_baselines.py --smoke-test
    python run_all_baselines.py --domain fire_ecology --workers 8
    python run_all_baselines.py --workers 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from baseline_parallel import missing_workspace_repos, resolve_workspace_root

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = resolve_workspace_root(_SCRIPT_DIR)

DOMAINS: dict[str, dict[str, str | int]] = {
    "coral_key": {
        "label": "Coral Key (ReefWatch)",
        "script": "Coral_Key_in_Three_Hour_Epochs/baselines/run_coral_key_baselines.py",
        "full_runs": 735,
    },
    "fire_ecology": {
        "label": "Fire Ecology",
        "script": "Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py",
        "full_runs": 2160,
    },
    "grain_guard": {
        "label": "Grain Guard",
        "script": "Xylella_SPQR/baselines/run_grain_guard_baselines.py",
        "full_runs": 324,
    },
}


def _run_domain(
    domain_key: str,
    *,
    smoke_test: bool,
    parallel: bool,
    workers: int | None,
) -> int:
    info = DOMAINS[domain_key]
    script = _WORKSPACE_ROOT / str(info["script"])
    if not script.is_file():
        print(f"[-] Runner not found: {script}")
        return 1

    cmd = [sys.executable, str(script)]
    if smoke_test:
        cmd.append("--smoke-test")
    if not parallel:
        cmd.append("--no-parallel")
    if workers is not None:
        cmd.extend(["--workers", str(workers)])

    n_runs = 1 if smoke_test else int(info["full_runs"])
    print("=" * 60)
    print(f"[*] {info['label']}: {n_runs} run(s)")
    print(f"[*] Command: {' '.join(cmd)}")
    print("=" * 60)

    return subprocess.call(cmd, cwd=_WORKSPACE_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline parameter scans across all domains (A0–A3, no BMA)"
    )
    parser.add_argument(
        "--domain",
        choices=[*DOMAINS.keys(), "all"],
        default="all",
        help="Domain to scan (default: all)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="One fast run per domain")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", action="store_false", dest="parallel")
    parser.add_argument("--workers", type=int, default=None, help="Worker process count")
    args = parser.parse_args(argv)

    missing = missing_workspace_repos(_WORKSPACE_ROOT)
    if missing:
        print(f"[-] Workspace missing sibling repos: {', '.join(missing)}")
        print("    Clone all repos under the workspace root and run .\\install_workspace.ps1")
        return 1

    domains = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]
    mode = "smoke test" if args.smoke_test else "full scan"
    total = sum(1 if args.smoke_test else int(DOMAINS[d]["full_runs"]) for d in domains)
    print(f"[*] Workspace: {_WORKSPACE_ROOT}")
    print(f"[*] Mode: {mode} | Domains: {', '.join(domains)} | Runs: {total}")

    exit_code = 0
    for domain_key in domains:
        rc = _run_domain(
            domain_key,
            smoke_test=args.smoke_test,
            parallel=args.parallel,
            workers=args.workers,
        )
        if rc != 0:
            print(f"[-] {DOMAINS[domain_key]['label']} failed (exit {rc})")
            exit_code = rc

    print("=" * 60)
    if exit_code == 0:
        print("[+] All baseline scans completed successfully.")
    else:
        print("[-] One or more baseline scans failed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
