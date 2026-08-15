"""Shared plumbing for the committed measurement scripts.

The measurement scripts all drive worlds through `scripts/run_ceiling_measurement.py`,
which lives outside the installed package, and all take the same population/seed
arguments. Loading and argument wiring live here so each measurement file contains only
its own measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_PATH = _REPO_ROOT / "scripts" / "run_ceiling_measurement.py"


def load_module(name: str, path: Path) -> ModuleType:
    """Import a module from a file path, registering it under `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_harness() -> ModuleType:
    """Load the committed ceiling harness for its world-construction helpers."""
    return load_module("run_ceiling_measurement", _HARNESS_PATH)


def add_shared_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the adapter, run-length, seed and population arguments every script takes."""
    parser.add_argument(
        "--adapter", default="tattletots.scenarios.sparse_sensor:SparseSensorScenario"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--grounded-fraction", type=float, default=0.67)
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    return parser


class Ledger(Protocol):
    """The observation interface every measurement ledger provides."""

    def observe(self, world: Any) -> None:
        """Fold one completed step of the world into the ledger."""

    def finalize(self, world: Any) -> None:
        """Read whatever is only available once the run is over."""


def drive_world(harness: ModuleType, world: Any, adapter: Any, steps: int, ledger: Ledger) -> None:
    """Advance adapter and world in lockstep, observing after each completed step."""
    for step in range(steps):
        adapter.step(step)
        active = adapter.get_active_locations(step)
        world.set_event_state(active)
        harness.set_oracle_locations(world, active)
        world.step()
        ledger.observe(world)
    ledger.finalize(world)


def harness_options(harness: ModuleType, args: argparse.Namespace) -> Any:
    """Build `HarnessOptions` from the shared arguments."""
    return harness.HarnessOptions(
        adapter_spec=args.adapter,
        steps=args.steps,
        seeds=tuple(args.seeds),
        initial_population=args.initial_population,
        max_population=args.max_population,
    )
