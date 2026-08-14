"""Domain-neutral conformance checks for published adapter instruments."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.config import SimulationConfig
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.interface.instrument import (
    InstrumentFinding,
    validate_stream_declarations,
)
from tattletots.models.location import EventLocation, LocationFrame
from tattletots.models.stream import Stream, StreamType


class AdapterConformanceCheck(StrEnum):
    """Checks reported by the adapter conformance suite."""

    EXECUTION = "execution"
    SENSOR_CALLS = "sensor_calls"
    STREAM_BYPASS = "stream_bypass"
    SILENCED_SENSOR_PROBE = "silenced_sensor_probe"
    DECODER_ROUND_TRIP = "decoder_round_trip"
    DECLARATIONS = "declarations"
    DECLARED_VS_DELIVERED = "declared_vs_delivered"
    STATE_INDEPENDENCE = "state_independence"
    FRAME_CONTAINMENT = "frame_containment"


@dataclass(frozen=True)
class AdapterConformanceFinding:
    """One structured adapter conformance result."""

    check: AdapterConformanceCheck
    passed: bool
    message: str
    measured: int | float | None = None
    threshold: int | float | None = None
    exercised: bool = True


@dataclass(frozen=True)
class AdapterConformanceReport:
    """Structured results from checking one adapter instance."""

    findings: tuple[AdapterConformanceFinding, ...]
    measured_steps: int
    sensor_inventory: tuple[str, ...]
    uncalled_sensors: tuple[str, ...]
    bypassed_streams: tuple[str, ...]
    decoder_failures: tuple[str, ...]
    state_independence_exercised: bool

    @property
    def valid(self) -> bool:
        """Whether every requested conformance check passed."""
        return all(finding.passed and finding.exercised for finding in self.findings)


StateIndependenceFactory = Callable[[], tuple[DomainAdapter, DomainAdapter]]


@dataclass(frozen=True)
class _SensorTarget:
    """One reflected sensor observation callable."""

    key: str
    instance_id: int
    owner: type[Any]
    method_name: str


@dataclass(frozen=True)
class _ReturnSpec:
    """Observed shape information used to silence one sensor method."""

    kind: str
    shape: tuple[int, ...] = ()


class _CallRecorder:
    """Record reflected sensor calls for each adapter step."""

    def __init__(self) -> None:
        self.called: set[str] = set()
        self.step_calls: set[str] = set()
        self.return_specs: dict[str, list[_ReturnSpec]] = {}

    def begin_step(self) -> None:
        self.step_calls = set()

    def record(self, key: str, value: Any = None) -> None:
        self.called.add(key)
        self.step_calls.add(key)
        spec = _return_spec(value)
        self.return_specs.setdefault(key, []).append(spec)


@contextmanager
def _record_sensor_calls(
    targets: tuple[_SensorTarget, ...],
    recorder: _CallRecorder,
    *,
    stub_returns: dict[str, tuple[Any, ...]] | None = None,
) -> Iterator[tuple[str, ...]]:
    """Temporarily wrap reflected sensor methods and record their calls."""
    grouped: dict[tuple[type[Any], str], dict[int, str]] = {}
    for target in targets:
        grouped.setdefault((target.owner, target.method_name), {})[target.instance_id] = target.key

    patches: list[tuple[type[Any], str, Any, bool]] = []
    tracking_errors: list[str] = []
    stub_indices: dict[str, int] = {}
    try:
        for (owner, method_name), instance_keys in grouped.items():
            raw = inspect.getattr_static(owner, method_name, None)
            if not inspect.isfunction(raw):
                tracking_errors.append(
                    f"{owner.__name__}.{method_name} is not a patchable Python method"
                )
                continue

            @wraps(raw)
            def wrapped(
                instance: Any,
                *args: Any,
                _raw: Any = raw,
                _instance_keys: dict[int, str] = instance_keys,
                _owner: type[Any] = owner,
                _method_name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                key = _instance_keys.get(
                    id(instance),
                    f"{_owner.__name__}.{_method_name} (unmapped instance)",
                )
                if stub_returns is not None and key in stub_returns:
                    index = stub_indices.get(key, 0)
                    analogues = stub_returns[key]
                    stub_indices[key] = index + 1
                    value = analogues[min(index, len(analogues) - 1)]
                    recorder.record(key, value)
                    return value
                value = _raw(instance, *args, **kwargs)
                recorder.record(key, value)
                return value

            had_own_method = method_name in owner.__dict__
            original = owner.__dict__.get(method_name)
            try:
                setattr(owner, method_name, wrapped)
            except (AttributeError, TypeError) as error:
                tracking_errors.append(
                    f"{owner.__name__}.{method_name} could not be wrapped: {error}"
                )
                continue
            patches.append((owner, method_name, original, had_own_method))
        yield tuple(tracking_errors)
    finally:
        for owner, method_name, original, had_own_method in reversed(patches):
            if had_own_method:
                setattr(owner, method_name, original)
            else:
                delattr(owner, method_name)


def validate_adapter_conformance(
    adapter: DomainAdapter,
    steps: int,
    *,
    state_independence_factory: StateIndependenceFactory | None = None,
    strict_state_independence: bool = False,
    config: SimulationConfig | None = None,
) -> AdapterConformanceReport:
    """Run domain-neutral publication, sensing, decoding, and state checks.

    Static sensor declarations are always required to be independent of hidden
    state.  Dynamic coordinates and statuses describe observations and may
    legitimately differ; those differences are reported informationally unless
    ``strict_state_independence`` is enabled.  Use the strict form for fixed
    coverage instruments whose dynamic fields are also expected to depend only
    on placement, cadence, and geometry.
    """
    if steps <= 0:
        return _invalid_steps_report(steps)

    streams = adapter.get_streams()
    targets = _find_sensor_targets(adapter)
    recorder = _CallRecorder()
    bypassed: set[str] = set()
    frame_failures: list[str] = []
    frame = adapter.get_location_frame()
    execution_error: str | None = None
    completed_steps = 0
    with _record_sensor_calls(targets, recorder) as tracking_errors:
        for time_step in range(steps):
            recorder.begin_step()
            before_streams = adapter.get_streams()
            before = _stream_snapshots(before_streams)
            try:
                adapter.step(time_step)
            except Exception as error:  # noqa: BLE001 - report adapter failures structurally
                execution_error = (
                    f"Adapter.step({time_step}) raised {type(error).__name__}: {error}"
                )
                break
            completed_steps += 1
            streams = adapter.get_streams()
            after = _stream_snapshots(streams)
            if frame is not None:
                try:
                    frame_failures.extend(
                        _outside_frame(
                            frame,
                            adapter.get_active_locations(time_step),
                            f"ground truth at step {time_step}",
                        )
                    )
                    report = adapter.infer_report_location(
                        [stream.current_data for stream in streams],
                        [stream.label for stream in streams],
                    )
                    if report is not None:
                        frame_failures.extend(
                            _outside_frame(frame, [report], f"decoded report at step {time_step}")
                        )
                except Exception as error:  # noqa: BLE001 - report frame checks structurally
                    frame_failures.append(
                        f"Frame check at step {time_step} raised {type(error).__name__}: {error}"
                    )
            if not recorder.step_calls:
                for label, before_data in before.items():
                    after_data = after.get(label)
                    if (
                        after_data is not None
                        and not np.array_equal(before_data, after_data)
                        and _is_raw_stream(before_streams + streams, label)
                    ):
                        bypassed.add(label)

    streams = adapter.get_streams()
    findings: list[AdapterConformanceFinding] = []
    if execution_error is None:
        findings.append(
            AdapterConformanceFinding(
                AdapterConformanceCheck.EXECUTION,
                True,
                f"Adapter stepped successfully for {completed_steps} steps.",
                measured=completed_steps,
                threshold=steps,
            )
        )
    else:
        findings.append(
            AdapterConformanceFinding(
                AdapterConformanceCheck.EXECUTION,
                False,
                execution_error,
            )
        )

    uncalled = tuple(target.key for target in targets if target.key not in recorder.called)
    sensor_message = (
        "Every reflected sensor observation method was called at least once."
        if not uncalled
        else "Never-called sensor objects: " + ", ".join(uncalled)
    )
    if tracking_errors:
        sensor_message += " Tracking limitations: " + "; ".join(tracking_errors)
    findings.append(
        AdapterConformanceFinding(
            AdapterConformanceCheck.SENSOR_CALLS,
            not uncalled and not tracking_errors,
            sensor_message,
            measured=len(recorder.called),
            threshold=len(targets),
        )
    )

    bypass_message = (
        "No published raw stream changed on a step without any sensor observation call."
        if not bypassed
        else "Raw streams changed without any sensor observation call on the same step: "
        + ", ".join(sorted(bypassed))
    )
    findings.append(
        AdapterConformanceFinding(
            AdapterConformanceCheck.STREAM_BYPASS,
            not bypassed,
            bypass_message
            + " This heuristic excludes derived, residual, and output streams; it does not "
            "prove which sensor should own a raw update.",
            measured=len(bypassed),
            threshold=0,
        )
    )

    declaration_findings = _declaration_findings(streams)
    declaration_failures = [finding for finding in declaration_findings if not finding.passed]
    findings.append(
        AdapterConformanceFinding(
            AdapterConformanceCheck.DECLARATIONS,
            not declaration_failures,
            (
                "Stream declaration checks reused validate_instrument's shared declaration "
                "validator; no failures were found."
                if not declaration_failures
                else "Published stream declarations are inconsistent: "
                + "; ".join(finding.message for finding in declaration_failures)
            ),
            measured=len(declaration_failures),
            threshold=0,
        )
    )

    delivered_findings = _declared_vs_delivered_findings(
        streams,
        (config or SimulationConfig()).max_stream_dim,
    )
    unreachable = [finding for finding in delivered_findings if not finding.passed]
    findings.append(
        AdapterConformanceFinding(
            AdapterConformanceCheck.DECLARED_VS_DELIVERED,
            not unreachable,
            (
                "Every declared stream feature and metadata entry is deliverable under "
                "the configured engine stream cap."
                if not unreachable
                else "Declared stream features exceed what agents receive: "
                + "; ".join(finding.message for finding in unreachable)
            ),
            measured=sum(int(finding.measured or 0) for finding in unreachable),
            threshold=0,
        )
    )

    decoder_failures = _decoder_round_trip_failures(adapter, streams)
    findings.append(
        AdapterConformanceFinding(
            AdapterConformanceCheck.DECODER_ROUND_TRIP,
            not decoder_failures,
            (
                "Every declared sensor coordinate round-tripped through the adapter decoder."
                if not decoder_failures
                else "Decoder round-trip failures: " + "; ".join(decoder_failures)
            ),
            measured=len(decoder_failures),
            threshold=0,
        )
    )

    state_finding, exercised = _state_independence_finding(
        state_independence_factory,
        steps,
        strict=strict_state_independence,
    )
    findings.append(state_finding)

    frame_finding = _frame_containment_finding(
        adapter,
        streams,
        frame_failures,
    )
    findings.append(frame_finding)

    # Runs last: silencing steps the adapter further and mutates published streams.
    findings.append(
        _silenced_sensor_probe(
            adapter,
            targets,
            recorder.return_specs,
            steps,
        )
    )

    return AdapterConformanceReport(
        findings=tuple(findings),
        measured_steps=steps,
        sensor_inventory=tuple(target.key for target in targets),
        uncalled_sensors=uncalled,
        bypassed_streams=tuple(sorted(bypassed)),
        decoder_failures=tuple(decoder_failures),
        state_independence_exercised=exercised,
    )


def assert_adapter_conformance(
    adapter: DomainAdapter,
    steps: int,
    *,
    state_independence_factory: StateIndependenceFactory | None = None,
    strict_state_independence: bool = False,
    config: SimulationConfig | None = None,
) -> AdapterConformanceReport:
    """Run conformance checks and raise a readable pytest assertion on failure."""
    report = validate_adapter_conformance(
        adapter,
        steps,
        state_independence_factory=state_independence_factory,
        strict_state_independence=strict_state_independence,
        config=config,
    )
    failures = [
        finding for finding in report.findings if not finding.passed or not finding.exercised
    ]
    if failures:
        details = "\n".join(f"- {finding.check.value}: {finding.message}" for finding in failures)
        raise AssertionError(f"Adapter conformance failed:\n{details}")
    return report


def _invalid_steps_report(steps: int) -> AdapterConformanceReport:
    finding = AdapterConformanceFinding(
        AdapterConformanceCheck.EXECUTION,
        False,
        "steps must be positive.",
    )
    return AdapterConformanceReport(
        findings=(finding,),
        measured_steps=steps,
        sensor_inventory=(),
        uncalled_sensors=(),
        bypassed_streams=(),
        decoder_failures=(),
        state_independence_exercised=False,
    )


def _return_spec(value: Any) -> _ReturnSpec:
    if value is None:
        return _ReturnSpec("none")
    if isinstance(value, np.ndarray):
        return _ReturnSpec("ndarray", value.shape)
    if isinstance(value, list):
        return _ReturnSpec("list")
    if isinstance(value, tuple):
        return _ReturnSpec("tuple")
    return _ReturnSpec("unsupported")


def _empty_analogue(spec: _ReturnSpec) -> Any:
    if spec.kind == "none":
        return None
    if spec.kind == "ndarray":
        return np.zeros(spec.shape, dtype=np.float64)
    if spec.kind == "list":
        return []
    if spec.kind == "tuple":
        return ()
    return None


def _silenced_sensor_probe(
    adapter: DomainAdapter,
    targets: tuple[_SensorTarget, ...],
    return_specs: dict[str, list[_ReturnSpec]],
    start_step: int,
) -> AdapterConformanceFinding:
    """Check whether raw streams vary while every reflected sensor is silenced."""
    if not targets:
        return AdapterConformanceFinding(
            AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
            True,
            "Not exercised: no reflected sensor methods returned a value to silence.",
            exercised=False,
        )
    if any(target.key not in return_specs for target in targets):
        missing = [target.key for target in targets if target.key not in return_specs]
        return AdapterConformanceFinding(
            AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
            True,
            "Not exercised: no observed return shape for " + ", ".join(missing) + ".",
            exercised=False,
        )
    if any(spec.kind == "unsupported" for specs in return_specs.values() for spec in specs):
        return AdapterConformanceFinding(
            AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
            True,
            "Not exercised: at least one reflected sensor returned an unsupported "
            "value shape; only None, lists, tuples, and ndarrays can be silenced.",
            exercised=False,
        )

    stubs = {
        key: tuple(_empty_analogue(spec) for spec in specs) for key, specs in return_specs.items()
    }
    recorder = _CallRecorder()
    varying_streams: set[str] = set()
    previous: dict[str, NDArray[np.float64]] | None = None
    try:
        with _record_sensor_calls(targets, recorder, stub_returns=stubs):
            for offset in range(2):
                adapter.step(start_step + offset)
                streams = adapter.get_streams()
                current = _stream_snapshots(streams)
                if previous is not None:
                    for label, before_data in previous.items():
                        after_data = current.get(label)
                        if (
                            after_data is not None
                            and not np.array_equal(before_data, after_data)
                            and _is_raw_stream(streams, label)
                        ):
                            varying_streams.add(label)
                previous = current
    except Exception as error:  # noqa: BLE001 - report probe limitations structurally
        return AdapterConformanceFinding(
            AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
            True,
            "Not exercised: adapter raised "
            f"{type(error).__name__} while sensor methods were silenced: {error}",
            exercised=False,
        )

    if varying_streams:
        return AdapterConformanceFinding(
            AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
            False,
            "Raw streams varied across consecutive steps while every reflected sensor "
            "was silenced: "
            + ", ".join(sorted(varying_streams))
            + ". This observes data surviving total sensor silence; it does not prove "
            "intent or identify the responsible internal state.",
            measured=len(varying_streams),
            threshold=0,
        )
    return AdapterConformanceFinding(
        AdapterConformanceCheck.SILENCED_SENSOR_PROBE,
        True,
        "No published raw stream varied across consecutive steps while every "
        "reflected sensor was silenced.",
        measured=0,
        threshold=0,
    )


def _find_sensor_targets(adapter: DomainAdapter) -> tuple[_SensorTarget, ...]:
    """Reflect objects exposing an observation-style callable."""
    targets: list[_SensorTarget] = []
    seen_objects: set[int] = set()
    seen_targets: set[tuple[int, str]] = set()
    stack: list[tuple[Any, str]] = [(adapter, "adapter")]
    while stack:
        value, path = stack.pop()
        object_id = id(value)
        if object_id in seen_objects or _is_leaf(value):
            continue
        seen_objects.add(object_id)
        if value is not adapter:
            for method_name in ("observe", "scan", "detect"):
                method = getattr(value, method_name, None)
                if not callable(method) or (object_id, method_name) in seen_targets:
                    continue
                owner = type(value)
                if inspect.isfunction(inspect.getattr_static(owner, method_name, None)):
                    key = f"{path}.{method_name}"
                    targets.append(_SensorTarget(key, object_id, owner, method_name))
                    seen_targets.add((object_id, method_name))

        for child_path, child in _children(value, path):
            stack.append((child, child_path))
    return tuple(targets)


def _children(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield f"{path}[{key!r}]", child
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            yield f"{path}[{index}]", child
        return
    try:
        attributes = vars(value)
    except TypeError:
        return
    for name, child in attributes.items():
        yield f"{path}.{name}", child


def _is_leaf(value: Any) -> bool:
    return isinstance(
        value,
        (str, bytes, bytearray, int, float, complex, bool, type(None), np.ndarray, type),
    )


def _stream_snapshots(streams: list[Stream]) -> dict[str, NDArray[np.float64]]:
    return {stream.label: np.array(stream.current_data, copy=True) for stream in streams}


def _is_raw_stream(streams: list[Stream], label: str) -> bool:
    return any(stream.label == label and stream.stream_type == StreamType.RAW for stream in streams)


def _declaration_findings(streams: list[Stream]) -> tuple[InstrumentFinding, ...]:
    findings: list[InstrumentFinding] = []
    for stream in streams:
        findings.extend(validate_stream_declarations(stream, 0))
    return tuple(findings)


def _declared_vs_delivered_findings(
    streams: list[Stream],
    max_stream_dim: int,
) -> tuple[AdapterConformanceFinding, ...]:
    findings: list[AdapterConformanceFinding] = []
    metadata_fields = (
        "coordinates",
        "sensor_coordinates",
        "modality",
        "identity",
        "footprints",
        "resolution",
    )
    for stream in streams:
        metadata = stream.metadata
        declared_lengths = [stream.dimensionality]
        if metadata is not None:
            declared_lengths.extend(
                len(values)
                for field in metadata_fields
                if (values := getattr(metadata, field)) is not None
            )
        declared_count = max(declared_lengths)
        delivered_count = min(stream.dimensionality, max_stream_dim)
        unreachable = max(0, declared_count - delivered_count)
        if unreachable:
            findings.append(
                AdapterConformanceFinding(
                    AdapterConformanceCheck.DECLARED_VS_DELIVERED,
                    False,
                    f"Stream '{stream.label}' declares {declared_count} features but "
                    f"agents receive {delivered_count}; {unreachable} declared "
                    "geometry/features cannot be received by agents.",
                    measured=unreachable,
                    threshold=0,
                )
            )
    return tuple(findings)


def _decoder_round_trip_failures(
    adapter: DomainAdapter,
    streams: list[Stream],
) -> list[str]:
    failures: list[str] = []
    for stream in streams:
        metadata = stream.metadata
        if metadata is None or metadata.sensor_coordinates is None:
            continue
        for index, coordinate in enumerate(metadata.sensor_coordinates):
            if coordinate is None or index >= stream.dimensionality:
                continue
            data = np.zeros(stream.dimensionality, dtype=np.float64)
            data[index] = 1.0
            expected = _coordinate_to_location(coordinate)
            if expected is None:
                failures.append(f"{stream.label}[{index}] declares an invalid coordinate")
                continue
            try:
                actual = adapter.infer_report_location([data], [stream.label])
            except Exception as error:  # noqa: BLE001 - report decoder failures structurally
                failures.append(f"{stream.label}[{index}] raised {type(error).__name__}: {error}")
                continue
            if actual != expected:
                failures.append(f"{stream.label}[{index}] expected {expected}, decoded {actual}")
    return failures


def _coordinate_to_location(
    coordinate: tuple[float, ...],
) -> EventLocation | None:
    if len(coordinate) < 2 or not np.all(np.isfinite(coordinate[:2])):
        return None
    return (int(round(coordinate[0])), int(round(coordinate[1])))


def _state_independence_finding(
    factory: StateIndependenceFactory | None,
    steps: int,
    *,
    strict: bool,
) -> tuple[AdapterConformanceFinding, bool]:
    if factory is None:
        return (
            AdapterConformanceFinding(
                AdapterConformanceCheck.STATE_INDEPENDENCE,
                False,
                "State-independence failed: supply state_independence_factory to compare "
                "identical sensor configurations across hidden states.",
            ),
            False,
        )
    try:
        first, second = factory()
        observation_differences: set[tuple[str, str]] = set()
        for time_step in range(steps):
            first.step(time_step)
            second.step(time_step)
            static_mismatch, dynamic_differences = _state_metadata_mismatch(first, second)
            if static_mismatch is not None:
                return (
                    AdapterConformanceFinding(
                        AdapterConformanceCheck.STATE_INDEPENDENCE,
                        False,
                        static_mismatch,
                    ),
                    True,
                )
            observation_differences.update(dynamic_differences)
    except Exception as error:  # noqa: BLE001 - report hook failures structurally
        return (
            AdapterConformanceFinding(
                AdapterConformanceCheck.STATE_INDEPENDENCE,
                False,
                f"State-independence hook failed with {type(error).__name__}: {error}",
            ),
            True,
        )
    if observation_differences:
        details = ", ".join(f"{label}.{field}" for label, field in sorted(observation_differences))
        message = (
            "Observation-dependent fields differed for "
            f"{details} across {steps} steps; this is informational because "
            "reported coordinates and statuses may legitimately depend on hidden "
            "observations."
        )
        if strict:
            message = (
                "Strict state-independence failed: "
                + message
                + " Pass strict_state_independence=False for mobile or "
                "self-reporting instruments."
            )
        return (
            AdapterConformanceFinding(
                AdapterConformanceCheck.STATE_INDEPENDENCE,
                not strict,
                message,
                measured=steps,
                threshold=steps,
            ),
            True,
        )
    return (
        AdapterConformanceFinding(
            AdapterConformanceCheck.STATE_INDEPENDENCE,
            True,
            f"Static sensor declarations, coordinates, and statuses matched across {steps} steps.",
            measured=steps,
            threshold=steps,
        ),
        True,
    )


def _state_metadata_mismatch(
    first: DomainAdapter,
    second: DomainAdapter,
) -> tuple[str | None, set[tuple[str, str]]]:
    first_streams = {stream.label: stream for stream in first.get_streams()}
    second_streams = {stream.label: stream for stream in second.get_streams()}
    if first_streams.keys() != second_streams.keys():
        return "State-independence streams differ by label.", set()
    observation_differences: set[tuple[str, str]] = set()
    for label, first_stream in first_streams.items():
        second_stream = second_streams[label]
        first_metadata = first_stream.metadata
        second_metadata = second_stream.metadata
        static_fields = (
            (
                "sensor_coordinates",
                None if first_metadata is None else first_metadata.sensor_coordinates,
                None if second_metadata is None else second_metadata.sensor_coordinates,
            ),
            (
                "footprints",
                None if first_metadata is None else first_metadata.footprints,
                None if second_metadata is None else second_metadata.footprints,
            ),
            (
                "resolution",
                None if first_metadata is None else first_metadata.resolution,
                None if second_metadata is None else second_metadata.resolution,
            ),
            (
                "modality",
                None if first_metadata is None else first_metadata.modality,
                None if second_metadata is None else second_metadata.modality,
            ),
        )
        for field_name, first_value, second_value in static_fields:
            if first_value != second_value:
                return (
                    f"Stream {label!r} {field_name} declaration changes with hidden state.",
                    set(),
                )
        first_coordinates = None if first_metadata is None else first_metadata.coordinates
        second_coordinates = None if second_metadata is None else second_metadata.coordinates
        if first_coordinates != second_coordinates:
            observation_differences.add((label, "coordinates"))
        first_identity = None if first_metadata is None else first_metadata.identity
        second_identity = None if second_metadata is None else second_metadata.identity
        if first_identity != second_identity:
            if first_coordinates != second_coordinates:
                observation_differences.add((label, "identity"))
            else:
                return (
                    f"Stream {label!r} identity declaration changes with hidden state.",
                    set(),
                )
        if not np.array_equal(first_stream.current_status, second_stream.current_status):
            observation_differences.add((label, "status"))
    return None, observation_differences


def _frame_containment_finding(
    adapter: DomainAdapter,
    streams: list[Stream],
    measured_failures: list[str],
) -> AdapterConformanceFinding:
    frame = adapter.get_location_frame()
    if frame is None:
        return AdapterConformanceFinding(
            AdapterConformanceCheck.FRAME_CONTAINMENT,
            False,
            "Frame containment failed: adapter does not declare a public location frame.",
        )
    failures = _declared_geometry_outside_frame(streams, frame) + measured_failures
    return AdapterConformanceFinding(
        AdapterConformanceCheck.FRAME_CONTAINMENT,
        not failures,
        (
            "Declared geometry, ground-truth locations, and decoded reports stayed "
            "inside the public frame."
            if not failures
            else "Locations outside the public frame: " + "; ".join(failures)
        ),
        measured=len(failures),
        threshold=0,
    )


def _declared_geometry_outside_frame(
    streams: list[Stream],
    frame: LocationFrame,
) -> list[str]:
    failures: list[str] = []
    for stream in streams:
        if stream.metadata is None:
            continue
        if stream.metadata.coordinates is not None:
            failures.extend(
                _outside_frame(
                    frame,
                    [
                        location
                        for coordinate in stream.metadata.coordinates
                        if coordinate
                        and (location := _coordinate_to_location(coordinate)) is not None
                    ],
                    f"{stream.label}.coordinates",
                )
            )
        if stream.metadata.sensor_coordinates is not None:
            failures.extend(
                _outside_frame(
                    frame,
                    [
                        location
                        for coordinate in stream.metadata.sensor_coordinates
                        if coordinate
                        and (location := _coordinate_to_location(coordinate)) is not None
                    ],
                    f"{stream.label}.sensor_coordinates",
                )
            )
    return failures


def _outside_frame(
    frame: LocationFrame,
    locations: list[EventLocation],
    label: str,
) -> list[str]:
    (lower_x, lower_y), (upper_x, upper_y) = frame
    return [
        f"{label} location {location}"
        for location in locations
        if not (lower_x <= location[0] <= upper_x and lower_y <= location[1] <= upper_y)
    ]
