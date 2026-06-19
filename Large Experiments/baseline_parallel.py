"""Multiprocess execution helpers for baseline parameter scan runners.

Baseline simulations are CPU-bound (NumPy + pure-Python domain loops). Thread
pools share one interpreter and contend on the GIL, so they rarely drive
multiple cores. ProcessPoolExecutor launches separate Python worker processes
that show up as distinct jobs in Task Manager and use CPU in parallel.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypeVar

R = TypeVar("R")


def resolve_workspace_root(start: Path | None = None) -> Path:
    """Walk up from start to find the sibling-repo workspace root."""
    anchor = (start or Path(__file__)).resolve()
    for parent in [anchor, *anchor.parents]:
        if (parent / "Coral_Key_in_Three_Hour_Epochs").is_dir() and (
            parent / "TattleTots"
        ).is_dir():
            return parent
    return anchor.parents[2] if len(anchor.parents) >= 3 else anchor


def resolve_worker_count(requested: int | None, n_jobs: int) -> int:
    """Pick a worker count capped by both CPU cores and pending jobs."""
    cpu = os.cpu_count() or 4
    if requested is not None:
        return max(1, min(requested, n_jobs))
    return max(1, min(cpu, n_jobs))


def run_process_pool(
    worker_fn: Callable[..., R],
    job_args: List[Tuple[Any, ...]],
    runs: List[Dict[str, Any]],
    *,
    max_workers: int,
    on_success: Callable[[Dict[str, Any], R], None],
    on_failure: Callable[[Dict[str, Any], Exception], None],
) -> None:
    """Execute jobs in a ProcessPoolExecutor with per-job progress logging."""
    total = len(job_args)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, *args): run
            for run, args in zip(runs, job_args, strict=True)
        }
        for future in as_completed(futures):
            run = futures[future]
            completed += 1
            name = run["name"]
            try:
                result = future.result()
                on_success(run, result)
                print(f"[+] Completed ({completed}/{total}): {name}")
            except Exception as exc:
                on_failure(run, exc)
                print(f"[-] Failed ({completed}/{total}): {name}: {exc}")
