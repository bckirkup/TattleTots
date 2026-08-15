"""Tests for domain-neutral instrument validity checks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from tattletots.interface.instrument import InstrumentCheck, validate_instrument
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario
from tattletots.scenarios.sparse_sensor import (
    MAX_GRID_SIZE,
    MAX_TOTAL_STEPS,
    SparseSensorScenario,
)


class _UninformativeDecoderSparse(SparseSensorScenario):
    def infer_report_location(
        self, _stream_data: list[NDArray[np.float64]], _stream_labels: list[str]
    ) -> tuple[int, int]:
        return (999, 999)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid_size": 0}, "grid_size"),
        ({"grid_size": MAX_GRID_SIZE + 1}, "grid_size"),
        ({"n_sensors": 0}, "n_sensors"),
        ({"n_sensors": 101}, "n_sensors"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_std": np.inf}, "noise_std"),
        ({"dropout_rate": -0.1}, "dropout_rate"),
        ({"dropout_rate": 1.1}, "dropout_rate"),
        ({"dropout_rate": np.nan}, "dropout_rate"),
        ({"decay_length": 0.0}, "decay_length"),
        ({"decay_length": np.inf}, "decay_length"),
        ({"total_steps": 0}, "total_steps"),
        ({"total_steps": MAX_TOTAL_STEPS + 1}, "total_steps"),
    ],
)
def test_sparse_sensor_rejects_invalid_configuration(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SparseSensorScenario(**overrides)


def test_sparse_sensor_default_configuration_steps() -> None:
    scenario = SparseSensorScenario()

    scenario.step(0)

    assert scenario.get_streams()[0].current_data.size == scenario.n_sensors
    assert scenario.get_active_locations(0)


def test_sparse_sensor_instrument_is_valid_and_reaches_above_chance() -> None:
    report = validate_instrument(SparseSensorScenario(seed=42), steps=200)

    assert report.valid
    assert report.event_steps == 200
    assert report.distinct_event_locations > 1
    assert report.inferability_precision > report.static_prior_baseline
    assert 0.0 <= report.decoder_precision <= 1.0
    assert 0.0 <= report.static_prior_baseline <= 1.0
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
    assert "does not carry" in inferability.message
    localization = next(
        finding for finding in report.findings if finding.check == InstrumentCheck.LOCALIZATION
    )
    assert not localization.passed
    assert "vacuous" in localization.message
    frame = next(
        finding for finding in report.findings if finding.check == InstrumentCheck.COORDINATE_FRAME
    )
    assert frame.passed
    assert 0.0 <= report.decoder_precision <= 1.0


def test_inferability_baseline_does_not_depend_on_decoder_output() -> None:
    reference = validate_instrument(SparseSensorScenario(seed=42), steps=40)
    uninformative = validate_instrument(_UninformativeDecoderSparse(seed=42), steps=40)

    assert uninformative.inferability_precision == reference.inferability_precision
    assert uninformative.chance_baseline == reference.chance_baseline
    assert uninformative.static_prior_baseline == reference.static_prior_baseline
    assert uninformative.decoder_precision != reference.decoder_precision


def test_instrument_report_exposes_structured_check_results() -> None:
    report = validate_instrument(SparseSensorScenario(seed=7), steps=40)

    assert {finding.check for finding in report.findings} >= {
        InstrumentCheck.EVENT_WINDOW,
        InstrumentCheck.COORDINATE_FRAME,
        InstrumentCheck.DECLARATIONS,
        InstrumentCheck.BASELINE,
        InstrumentCheck.LOCALIZATION,
        InstrumentCheck.INFERABILITY,
    }
    assert all(finding.message for finding in report.findings)
