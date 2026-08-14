"""Tests for domain-neutral adapter conformance checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tattletots.interface.adapter_conformance import (
    AdapterConformanceCheck,
    assert_adapter_conformance,
    validate_adapter_conformance,
)
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.dispatch_target import DispatchTarget
from tattletots.models.location import EventLocation
from tattletots.models.observation import ObservationStatus, StreamMetadata
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User


@dataclass
class _Probe:
    """Minimal sensor-like object for reflection tests."""

    def observe(self, value: float) -> NDArray[np.float64]:
        return np.array([value], dtype=np.float64)


class _FixtureAdapter(DomainAdapter):
    """Configurable fake adapter with one published sensor feature."""

    def __init__(
        self,
        *,
        hidden_state: bool = False,
        leak_status: bool = False,
        wrong_decoder: bool = False,
        bypass_stream: bool = False,
        unpublished_sensor: bool = False,
    ) -> None:
        self._probe = _Probe()
        self._unpublished_probe = _Probe() if unpublished_sensor else None
        self._hidden_state = hidden_state
        self._leak_status = leak_status
        self._wrong_decoder = wrong_decoder
        self._bypass_stream = bypass_stream
        self._stream = Stream(
            stream_type=StreamType.RAW,
            dimensionality=1,
            label="published_signal",
            current_data=np.zeros(1, dtype=np.float64),
            current_status=np.array([ObservationStatus.OBSERVED.value]),
            metadata=StreamMetadata(
                coordinates=[(1.0, 1.0)],
                sensor_coordinates=[(1.0, 1.0)],
                modality=["signal"],
                identity=[None],
                footprints=[(1.0, 1.0)],
                resolution=[1.0],
            ),
        )

    def get_streams(self) -> list[Stream]:
        return [self._stream]

    def get_users(self) -> list[User]:
        return []

    def step(self, time_step: int) -> None:
        if not self._bypass_stream or time_step == 0:
            data = self._probe.observe(float(time_step + 1))
        else:
            data = np.array([float(time_step + 1)], dtype=np.float64)
        status = (
            ObservationStatus.MISSING.value
            if self._leak_status and self._hidden_state
            else ObservationStatus.OBSERVED.value
        )
        self._stream.update(data, [status])

    def get_ground_truth(self, time_step: int) -> bool:
        return True

    def get_active_locations(self, time_step: int) -> list[EventLocation]:
        return [(1, 1)]

    def get_location_frame(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((0, 0), (2, 2))

    def infer_report_location(
        self,
        stream_data: list[NDArray[np.float64]],
        stream_labels: list[str],
    ) -> EventLocation:
        if self._wrong_decoder:
            return (0, 0)
        return (1, 1)

    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        return 0.0

    def compute_costs(
        self,
        n_escalations: int,
        n_correct: int,
        n_false_alarms: int,
        n_missed: int,
    ) -> dict[str, float]:
        return {
            "surveillance_cost": 0.0,
            "response_cost": 0.0,
            "damage_cost": 0.0,
        }

    def get_responder_user_id(self) -> str:
        return ""

    def dispatch_and_judge_responses(
        self,
        targets: list[DispatchTarget],
        time_step: int,
    ) -> list[ResponseOutcome]:
        return []


class _NestedFixtureAdapter(_FixtureAdapter):
    """Place sensor-like objects in both supported container forms."""

    def __init__(self) -> None:
        super().__init__()
        self._nested = [_Probe()]
        self._mapping = {"secondary": _Probe()}

    def step(self, time_step: int) -> None:
        super().step(time_step)
        self._nested[0].observe(float(time_step))
        self._mapping["secondary"].observe(float(time_step))


class _ResidualFixtureAdapter(_FixtureAdapter):
    """Change an engine-derived stream on a sensor-free step."""

    def __init__(self) -> None:
        super().__init__()
        self._residual = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=1,
            label="derived_signal",
            current_data=np.zeros(1, dtype=np.float64),
        )

    def get_streams(self) -> list[Stream]:
        return [self._stream, self._residual]

    def step(self, time_step: int) -> None:
        if time_step == 0:
            super().step(time_step)
        else:
            self._residual.update(np.array([float(time_step)], dtype=np.float64))


class _MalformedDeclarationAdapter(_FixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._stream.metadata.modality = []


class _OutOfFrameAdapter(_FixtureAdapter):
    def get_location_frame(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((0, 0), (0, 0))


def _matching_state_factory() -> tuple[DomainAdapter, DomainAdapter]:
    return (
        _FixtureAdapter(),
        _FixtureAdapter(hidden_state=True),
    )


def _matching_residual_state_factory() -> tuple[DomainAdapter, DomainAdapter]:
    return _ResidualFixtureAdapter(), _ResidualFixtureAdapter()


def _matching_malformed_state_factory() -> tuple[DomainAdapter, DomainAdapter]:
    return _MalformedDeclarationAdapter(), _MalformedDeclarationAdapter()


def _matching_out_of_frame_state_factory() -> tuple[DomainAdapter, DomainAdapter]:
    return _OutOfFrameAdapter(), _OutOfFrameAdapter()


def _leaking_state_factory() -> tuple[DomainAdapter, DomainAdapter]:
    return (
        _FixtureAdapter(),
        _FixtureAdapter(hidden_state=True, leak_status=True),
    )


def _finding_checks(report) -> set[AdapterConformanceCheck]:
    return {finding.check for finding in report.findings if not finding.passed}


def test_conformant_fixture_passes_all_checks() -> None:
    report = assert_adapter_conformance(
        _FixtureAdapter(),
        steps=3,
        state_independence_factory=_matching_state_factory,
    )

    assert report.valid
    assert report.uncalled_sensors == ()
    assert report.bypassed_streams == ()


def test_sensor_never_published_is_caught_by_sensor_call_check() -> None:
    report = validate_adapter_conformance(
        _FixtureAdapter(unpublished_sensor=True),
        steps=3,
        state_independence_factory=_matching_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.SENSOR_CALLS}
    assert any("unpublished_probe" in sensor for sensor in report.uncalled_sensors)


def test_raw_stream_bypass_is_caught_without_sensor_call() -> None:
    report = validate_adapter_conformance(
        _FixtureAdapter(bypass_stream=True),
        steps=3,
        state_independence_factory=_matching_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.STREAM_BYPASS}
    assert report.bypassed_streams == ("published_signal",)


def test_wrong_field_arithmetic_decoder_is_caught_by_round_trip() -> None:
    report = validate_adapter_conformance(
        _FixtureAdapter(wrong_decoder=True),
        steps=3,
        state_independence_factory=_matching_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.DECODER_ROUND_TRIP}
    assert report.decoder_failures


def test_state_dependent_status_is_caught_by_state_check() -> None:
    report = validate_adapter_conformance(
        _FixtureAdapter(leak_status=True),
        steps=3,
        state_independence_factory=_leaking_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.STATE_INDEPENDENCE}


def test_missing_state_hook_is_explicitly_not_exercised() -> None:
    report = validate_adapter_conformance(_FixtureAdapter(), steps=1)

    finding = next(
        finding
        for finding in report.findings
        if finding.check == AdapterConformanceCheck.STATE_INDEPENDENCE
    )
    assert not finding.passed
    assert "Not exercised" in finding.message


def test_reflection_finds_sensors_in_nested_lists_and_dicts() -> None:
    report = validate_adapter_conformance(
        _NestedFixtureAdapter(),
        steps=2,
        state_independence_factory=_matching_state_factory,
    )

    assert report.valid
    assert len(report.sensor_inventory) == 3
    assert any("_nested" in sensor for sensor in report.sensor_inventory)
    assert any("_mapping" in sensor for sensor in report.sensor_inventory)


def test_wrapped_sensor_methods_are_restored_after_validation() -> None:
    original = _Probe.observe

    report = validate_adapter_conformance(
        _FixtureAdapter(),
        steps=2,
        state_independence_factory=_matching_state_factory,
    )

    assert report.valid
    assert _Probe.observe is original


def test_residual_stream_changes_are_excluded_from_raw_bypass_check() -> None:
    report = validate_adapter_conformance(
        _ResidualFixtureAdapter(),
        steps=2,
        state_independence_factory=_matching_residual_state_factory,
    )

    assert report.valid
    assert report.bypassed_streams == ()


def test_declaration_consistency_failure_is_reported_structurally() -> None:
    report = validate_adapter_conformance(
        _MalformedDeclarationAdapter(),
        steps=1,
        state_independence_factory=_matching_malformed_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.DECLARATIONS}


def test_frame_containment_checks_declared_and_measured_locations() -> None:
    report = validate_adapter_conformance(
        _OutOfFrameAdapter(),
        steps=1,
        state_independence_factory=_matching_out_of_frame_state_factory,
    )

    assert _finding_checks(report) == {AdapterConformanceCheck.FRAME_CONTAINMENT}


def test_assertion_wrapper_includes_failed_check_details() -> None:
    try:
        assert_adapter_conformance(
            _FixtureAdapter(wrong_decoder=True),
            steps=1,
            state_independence_factory=_matching_state_factory,
        )
    except AssertionError as error:
        assert "decoder_round_trip" in str(error)
    else:
        raise AssertionError("Expected the conformance assertion to fail")
