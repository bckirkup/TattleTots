"""Tests for the committed ceiling/instrument measurement harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_ceiling_measurement.py"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ceiling_measurement", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load harness from {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _options(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "steps": 25,
        "seeds": (42,),
        "grounded_fractions": (0.0,),
        "grounded_multipliers": (1.0,),
        "initial_population": 10,
        "max_population": 20,
        "arms": ("ordinary",),
    }
    defaults.update(overrides)
    return harness.HarnessOptions(**defaults)


def test_default_adapter_instrument_is_non_vacuous() -> None:
    nulls = harness.instrument_nulls(_options())

    assert nulls["distinct_event_locations"] > 1
    assert 0.0 < nulls["static_prior_baseline"] < 0.99
    assert nulls["chance_baseline"] > 0.0
    assert nulls["inferability_precision"] > nulls["chance_baseline"]


@pytest.mark.parametrize("arm", ["ordinary", "oracle_monoculture", "oracle_invasion"])
def test_grid_point_metrics_stay_in_bounds(arm: str) -> None:
    point = harness.GridPoint(arm=arm, grounded_fraction=0.34, grounded_multiplier=2.0)
    metrics, output = harness.measure_grid_point(point, seed=42, options=_options())

    for key in (
        "correct_report_rate",
        "per_capita_attention_solvency",
        "grounded_yield_share",
        "effective_grounded_yield_share",
    ):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of bounds: {metrics[key]}"
    assert -1.0 <= metrics["parent_child_reproductive_correlation"] <= 1.0
    assert output.run_summary.domain == "SparseSensorScenario"
    assert output.ecology_metrics.total_reports == metrics["total_reports"]


def test_grounded_fraction_raises_measured_grounded_yield_share() -> None:
    shares = []
    for fraction in (0.0, 0.5, 1.0):
        point = harness.GridPoint(
            arm="ordinary", grounded_fraction=fraction, grounded_multiplier=1.0
        )
        metrics, _ = harness.measure_grid_point(point, seed=42, options=_options())
        shares.append(metrics["grounded_yield_share"])

    assert shares == sorted(shares), f"grounded-yield share is not monotone: {shares}"
    assert shares[-1] - shares[0] > 0.1, f"reservation barely moved grounded yield: {shares}"


def test_falsification_verdict_reads_the_measured_grid() -> None:
    results = harness.run_measurement(_options(grounded_fractions=(0.0, 1.0)))
    verdict = harness.falsification_verdict(results)

    assert set(results["summary"]) == {
        "ordinary|fraction=0|multiplier=1",
        "ordinary|fraction=1|multiplier=1",
    }
    assert verdict["static_prior_null"] == pytest.approx(
        results["instrument"]["static_prior_baseline"]
    )
    assert isinstance(verdict["passed"], bool)
    assert harness.markdown_report(results, verdict).startswith("# Ceiling measurement")


def test_adapter_spec_must_be_module_and_callable() -> None:
    with pytest.raises(ValueError, match="module:Callable"):
        harness.build_adapter("tattletots.scenarios.sparse_sensor", seed=1, steps=5)
