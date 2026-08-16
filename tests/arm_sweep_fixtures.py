"""Shared fixtures for the config-arm sweep measurement scripts.

Every lever's sweep script takes the same shared arguments and renders its arms through
`config_arm_sweep.markdown_report`, so their tests build the same argument namespace and
the same pooled-results skeleton. These builders keep that shape in one place; each test
module adds only the arguments and arms specific to its lever.
"""

from __future__ import annotations

import argparse
from typing import Any

ADAPTER = "tattletots.scenarios.sparse_sensor:SparseSensorScenario"


def sweep_args(**overrides: Any) -> argparse.Namespace:
    """Arguments every sweep script shares, before its own lever-specific ones."""
    defaults: dict[str, Any] = {
        "adapter": ADAPTER,
        "arm": "ordinary",
        "steps": 40,
        "seeds": [42],
        "grounded_fraction": 0.67,
        "initial_population": 20,
        "max_population": 60,
        "correct_report_value": 8.0,
        "break_even_precision": 0.2,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def arm(label: str, **summary: float) -> tuple[str, dict[str, Any]]:
    """One pooled arm entry as the sweep runner emits it."""
    return label, {"config": {}, "summary": {"n_runs": 3, **summary}, "runs": []}


def empty_arm() -> dict[str, Any]:
    """An arm whose runs all failed to score, which the renderer must mark rather than zero."""
    return {"config": {}, "summary": {"n_runs": 0}, "runs": []}


def sweep_results(arms: dict[str, dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Pooled sweep results with the keys `config_arm_sweep.markdown_report` reads."""
    results: dict[str, Any] = {
        "adapter": ADAPTER,
        "arm": "ordinary",
        "steps": 200,
        "seeds": [42, 43],
        "grounded_input_fraction": 0.67,
        "max_population": 60,
        "correct_report_attention_value": 8.0,
        "break_even_precision": 0.2,
        "arms": arms,
    }
    results.update(overrides)
    return results
