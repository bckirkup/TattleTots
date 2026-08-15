"""Tests for the correctness-heritability measurement script."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from tattletots.telemetry.payoff_ledger import AgentPayoffRecord

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "measure_correctness_heritability.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_correctness_heritability", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load script from {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


@dataclass
class _StubLedger:
    """Ledger stand-in exposing only the `records` view the measurements read."""

    records: list[AgentPayoffRecord] = field(default_factory=list)


def _ledger(records: list[AgentPayoffRecord]) -> _StubLedger:
    return _StubLedger(records=records)


def _record(
    agent_id: str,
    reports: int,
    correct: int,
    parent_ids: tuple[str, ...] = (),
) -> AgentPayoffRecord:
    record = AgentPayoffRecord(agent_id=agent_id, parent_ids=parent_ids)
    record.reports_issued = reports
    record.correct_reports = correct
    record.adult_steps = 10
    return record


def test_report_count_conditioning_drops_pairs_below_the_threshold() -> None:
    """Conditioning on report count only keeps pairs where both sides clear it."""
    ledger = _ledger(
        [
            _record("p1", reports=8, correct=4),
            _record("c1", reports=8, correct=4, parent_ids=("p1",)),
            _record("p2", reports=1, correct=0),
            _record("c2", reports=1, correct=1, parent_ids=("p2",)),
        ]
    )
    conditioned = script.report_count_conditioning([ledger], [1, 5])
    assert conditioned[1][1] == 2
    assert conditioned[5][1] == 1


def test_excess_variance_ratio_separates_noise_from_real_spread() -> None:
    """A pure coin-flip cohort sits near 1.0; a split cohort sits well above it."""
    noise_only = _ledger([_record(f"a{i}", reports=1, correct=i % 2) for i in range(40)])
    real_spread = _ledger(
        [
            _record(f"lo{i}", reports=20, correct=0)
            if i % 2
            else _record(f"hi{i}", reports=20, correct=20)
            for i in range(40)
        ]
    )
    noisy = script.excess_variance([noise_only], min_reports=1)["excess_variance_ratio"]
    structured = script.excess_variance([real_spread], min_reports=1)["excess_variance_ratio"]
    assert noisy == pytest.approx(1.0, abs=0.2)
    assert structured > 5.0 * noisy


def test_reporting_opportunity_summarises_per_agent_sample_size() -> None:
    """Reported opportunity tracks the reports an agent's precision estimate rests on."""
    ledger = _ledger([_record("a", reports=0, correct=0), _record("b", reports=6, correct=3)])
    opportunity = script.reporting_opportunity([ledger])
    assert opportunity["n_adults"] == pytest.approx(2.0)
    assert opportunity["mean_reports_per_adult"] == pytest.approx(3.0)
    assert opportunity["share_with_at_least_5_reports"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("reports", "expected_order"),
    [(0.5, 0), (5.0, 1), (50.0, 2)],
)
def test_attenuation_rises_with_reports_per_agent(reports: float, expected_order: int) -> None:
    """More reports per agent means less attenuation of a heritable trait."""
    ladder = [
        script.attenuation_prediction(0.015, 0.12, n, 0.63)["attenuation_factor"]
        for n in (0.5, 5.0, 50.0)
    ]
    assert ladder[0] < ladder[1] < ladder[2] <= 1.0
    prediction = script.attenuation_prediction(0.015, 0.12, reports, 0.63)
    assert prediction["attenuation_factor"] == pytest.approx(ladder[expected_order])
    assert 0.0 <= prediction["predicted_parent_child_precision_r"] <= 0.63


def test_attenuation_prediction_is_empty_without_genomic_variance() -> None:
    """With no genomic variance there is no heritable signal to attenuate."""
    assert script.attenuation_prediction(0.0, 0.12, 5.0, 0.63) == {}
