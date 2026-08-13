"""Tests for domain-neutral instrument validity checks."""

from __future__ import annotations

from tattletots.interface.instrument import InstrumentCheck, validate_instrument
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario
from tattletots.scenarios.sparse_sensor import SparseSensorScenario


def test_sparse_sensor_instrument_is_valid_and_reaches_above_chance() -> None:
    report = validate_instrument(SparseSensorScenario(seed=42), steps=200)

    assert report.valid
    assert report.event_steps == 200
    assert report.distinct_event_locations > 1
    assert report.inferability_precision > report.chance_baseline
    assert all(
        finding.passed
        for finding in report.findings
        if finding.check == InstrumentCheck.DECLARATIONS
    )


def test_gaussian_shift_instrument_rejects_unobservable_location_label() -> None:
    report = validate_instrument(GaussianShiftScenario(seed=42), steps=200)

    assert not report.valid
    inferability = next(
        finding for finding in report.findings if finding.check == InstrumentCheck.INFERABILITY
    )
    assert not inferability.passed
    assert "does not recover" in inferability.message


def test_instrument_report_exposes_structured_check_results() -> None:
    report = validate_instrument(SparseSensorScenario(seed=7), steps=40)

    assert {finding.check for finding in report.findings} >= {
        InstrumentCheck.EVENT_WINDOW,
        InstrumentCheck.COORDINATE_FRAME,
        InstrumentCheck.DECLARATIONS,
        InstrumentCheck.INFERABILITY,
    }
    assert all(finding.message for finding in report.findings)
