"""Domain-neutral instrument validity checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.location import EventLocation, LocationFrame
from tattletots.models.observation import ObservationStatus, StreamMetadata
from tattletots.models.stream import Stream
from tattletots.telemetry.spatial_nulls import static_prior_precision


class InstrumentCheck(StrEnum):
    """Validity dimensions reported by the instrument validator."""

    EVENT_WINDOW = "event_window"
    COORDINATE_FRAME = "coordinate_frame"
    DECLARATIONS = "declarations"
    INFERABILITY = "inferability"
    BASELINE = "baseline"
    LOCALIZATION = "localization"


@dataclass(frozen=True)
class InstrumentFinding:
    """One structured instrument-validity finding."""

    check: InstrumentCheck
    passed: bool
    message: str
    measured: float | int | None = None
    threshold: float | int | None = None


@dataclass(frozen=True)
class InstrumentValidityReport:
    """Results from validating a domain adapter over a measured window."""

    findings: tuple[InstrumentFinding, ...]
    measured_steps: int
    event_steps: int
    distinct_event_locations: int
    inferability_precision: float
    decoder_precision: float
    chance_baseline: float
    static_prior_baseline: float
    candidate_locations: tuple[EventLocation, ...]

    @property
    def valid(self) -> bool:
        """Whether every instrument check passed."""
        return all(finding.passed for finding in self.findings)


def validate_instrument(
    adapter: DomainAdapter,
    steps: int,
    *,
    inferability_margin: float = 0.02,
) -> InstrumentValidityReport:
    """Validate an adapter's measurement window and published evidence.

    The adapter is stepped exactly once per measured time step. Validation
    consumes the adapter's run; callers should construct a fresh adapter for
    each validation or subsequent measurement. Evidence support is measured
    from declared coordinates and the adapter's public dimension mapping, not
    from the incumbent report decoder. Decoder precision is retained as an
    informational comparison. The validator does not inspect domain internals
    or compare against expected golden values.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    frame = adapter.get_location_frame()
    event_steps = 0
    event_locations: set[EventLocation] = set()
    evidence_locations: set[EventLocation] = set()
    correct_reports = 0
    supported_events = 0
    reportable_events = 0
    findings: list[InstrumentFinding] = []
    active_location_history: list[tuple[tuple[EventLocation, ...], int]] = []

    for time_step in range(steps):
        adapter.step(time_step)
        active = tuple(adapter.get_active_locations(time_step))
        is_event = adapter.get_ground_truth(time_step)
        active_location_history.append((active, int(is_event)))
        if is_event:
            event_steps += 1
            event_locations.update(active)
            reportable_events += 1

        stream_data = []
        stream_labels = []
        for stream in adapter.get_streams():
            _validate_stream_declarations(stream, time_step, findings)
            stream_data.append(stream.current_data)
            stream_labels.append(stream.label)
            _collect_stream_coordinates(stream.metadata, evidence_locations)

        if is_event and any(location in evidence_locations for location in active):
            supported_events += 1
        report = adapter.infer_report_location(stream_data, stream_labels)
        if is_event and report in active:
            correct_reports += 1

        if frame is not None:
            _check_frame_location(frame, active, "ground truth", time_step, findings)
            _check_frame_location(frame, [report], "report", time_step, findings)

    candidates = _candidate_locations(
        frame,
        evidence_locations,
        adapter,
    )
    chance_baseline = 1.0 / len(candidates) if candidates else 1.0
    static_prior_baseline = static_prior_precision(active_location_history)
    support_precision = supported_events / reportable_events if reportable_events else 0.0
    decoder_precision = correct_reports / reportable_events if reportable_events else 0.0

    _append_window_finding(findings, steps, event_steps, event_locations)
    if frame is None:
        findings.append(
            InstrumentFinding(
                check=InstrumentCheck.COORDINATE_FRAME,
                passed=True,
                message="No public frame declared; legacy coordinate behavior applies.",
            )
        )
    else:
        frame_failures = [
            finding
            for finding in findings
            if finding.check == InstrumentCheck.COORDINATE_FRAME and not finding.passed
        ]
        findings.append(
            InstrumentFinding(
                check=InstrumentCheck.COORDINATE_FRAME,
                passed=not frame_failures,
                message=(
                    "Ground-truth and report locations stay within the declared frame."
                    if not frame_failures
                    else "Locations fall outside the declared grading frame."
                ),
            )
        )
    findings.append(
        InstrumentFinding(
            check=InstrumentCheck.BASELINE,
            passed=True,
            message=(
                "Static-prior precision is "
                f"{static_prior_baseline:.2%} versus uniform precision "
                f"{chance_baseline:.2%}; static prior is the localization "
                "competence null, while uniform is the inferability null."
            ),
            measured=static_prior_baseline,
            threshold=chance_baseline,
        )
    )
    localization_vacuous = len(event_locations) < 2 or static_prior_baseline >= 0.99
    findings.append(
        InstrumentFinding(
            check=InstrumentCheck.LOCALIZATION,
            passed=not localization_vacuous,
            message=(
                "Localization is non-vacuous across multiple event locations."
                if not localization_vacuous
                else "Localization is vacuous because event locations have no meaningful spread."
            ),
            measured=static_prior_baseline,
            threshold=0.99,
        )
    )
    findings.append(
        InstrumentFinding(
            check=InstrumentCheck.INFERABILITY,
            passed=reportable_events > 0
            and support_precision > chance_baseline + inferability_margin,
            message=(
                "Published evidence carries event locations above uniform chance."
                if reportable_events > 0
                and support_precision > chance_baseline + inferability_margin
                else "Published evidence does not carry event locations above uniform chance."
            ),
            measured=support_precision,
            threshold=chance_baseline + inferability_margin,
        )
    )

    return InstrumentValidityReport(
        findings=_summarize_declarations(findings),
        measured_steps=steps,
        event_steps=event_steps,
        distinct_event_locations=len(event_locations),
        inferability_precision=support_precision,
        decoder_precision=decoder_precision,
        chance_baseline=chance_baseline,
        static_prior_baseline=static_prior_baseline,
        candidate_locations=tuple(sorted(candidates)),
    )


def _validate_stream_declarations(
    stream: Stream, time_step: int, findings: list[InstrumentFinding]
) -> None:
    """Validate lengths and status/metadata consistency for one stream."""
    dimensionality = stream.current_data.size
    status = stream.current_status
    metadata = stream.metadata
    if status.size not in (0, dimensionality):
        findings.append(
            InstrumentFinding(
                InstrumentCheck.DECLARATIONS,
                False,
                f"Stream {stream.label!r} has status length {status.size}, "
                f"data length {dimensionality} at step {time_step}.",
            )
        )
    if metadata is not None:
        try:
            metadata.validate_dimensionality(dimensionality)
        except ValueError as error:
            findings.append(InstrumentFinding(InstrumentCheck.DECLARATIONS, False, str(error)))
    if status.size != dimensionality or metadata is None:
        return
    for index, state in enumerate(status):
        if state not in {item.value for item in ObservationStatus}:
            findings.append(
                InstrumentFinding(
                    InstrumentCheck.DECLARATIONS,
                    False,
                    f"Unknown observation status {state!r} in stream {stream.label!r}.",
                )
            )
        if state != ObservationStatus.MISSING.value:
            continue
        coordinates = metadata.coordinates
        identity = metadata.identity
        if coordinates is not None and coordinates[index] is not None:
            findings.append(
                InstrumentFinding(
                    InstrumentCheck.DECLARATIONS,
                    False,
                    f"Missing feature {index} in stream {stream.label!r} "
                    "declares observed-object coordinates.",
                )
            )
        if identity is not None and identity[index] is not None:
            findings.append(
                InstrumentFinding(
                    InstrumentCheck.DECLARATIONS,
                    False,
                    f"Missing feature {index} in stream {stream.label!r} "
                    "declares observed-object identity.",
                )
            )
        sensor_coordinates = metadata.sensor_coordinates
        if sensor_coordinates is None or sensor_coordinates[index] is None:
            findings.append(
                InstrumentFinding(
                    InstrumentCheck.DECLARATIONS,
                    True,
                    f"Missing feature {index} in stream {stream.label!r} "
                    "has no declared static sensor geometry; its absence is less localizable.",
                )
            )


def _collect_stream_coordinates(
    metadata: StreamMetadata | None, locations: set[EventLocation]
) -> None:
    """Collect finite integer-valued coordinate declarations."""
    if metadata is None:
        return
    for coordinate_values in (metadata.coordinates, metadata.sensor_coordinates):
        if coordinate_values is None:
            continue
        for coordinate in coordinate_values:
            if coordinate is not None and len(coordinate) >= 2 and np.all(np.isfinite(coordinate)):
                locations.add((int(round(coordinate[0])), int(round(coordinate[1]))))


def _candidate_locations(
    frame: LocationFrame | None,
    evidence_locations: set[EventLocation],
    adapter: DomainAdapter,
) -> set[EventLocation]:
    """Build a public candidate set without reading domain internals."""
    if frame is not None:
        (lower_x, lower_y), (upper_x, upper_y) = frame
        return {(x, y) for x in range(lower_x, upper_x + 1) for y in range(lower_y, upper_y + 1)}
    candidates = set(evidence_locations)
    offset = 0
    for stream in adapter.get_streams():
        for index in range(stream.current_data.size):
            candidates.add(adapter.dim_index_to_location(offset + index))
        offset += stream.current_data.size
    return candidates


def _check_frame_location(
    frame: LocationFrame,
    locations: list[EventLocation] | tuple[EventLocation, ...],
    label: str,
    time_step: int,
    findings: list[InstrumentFinding],
) -> None:
    """Record any location outside an inclusive declared frame."""
    (lower_x, lower_y), (upper_x, upper_y) = frame
    for location in locations:
        if not (lower_x <= location[0] <= upper_x and lower_y <= location[1] <= upper_y):
            findings.append(
                InstrumentFinding(
                    InstrumentCheck.COORDINATE_FRAME,
                    False,
                    f"{label} location {location} is outside the frame at step {time_step}.",
                )
            )


def _append_window_finding(
    findings: list[InstrumentFinding],
    steps: int,
    event_steps: int,
    event_locations: set[EventLocation],
) -> None:
    """Record whether the measured window contains a useful event task."""
    passed = event_steps > 0 and (event_steps < steps or len(event_locations) > 1)
    if event_steps == 0:
        message = "No ground-truth events occur in the measured window."
    elif event_steps == steps and len(event_locations) <= 1:
        message = "Every measured step has the same event state and location."
    else:
        message = "The measured window contains observable event variation."
    findings.append(
        InstrumentFinding(
            InstrumentCheck.EVENT_WINDOW,
            passed,
            message,
            measured=event_steps,
            threshold=1,
        )
    )


def _summarize_declarations(
    findings: list[InstrumentFinding],
) -> tuple[InstrumentFinding, ...]:
    """Add a single declaration summary while preserving detailed findings."""
    unique_findings: list[InstrumentFinding] = []
    seen: set[tuple[InstrumentCheck, str]] = set()
    for finding in findings:
        key = (finding.check, finding.message)
        if key not in seen:
            unique_findings.append(finding)
            seen.add(key)
    findings[:] = unique_findings
    declaration_failures = [
        finding
        for finding in findings
        if finding.check == InstrumentCheck.DECLARATIONS and not finding.passed
    ]
    findings.append(
        InstrumentFinding(
            InstrumentCheck.DECLARATIONS,
            not declaration_failures,
            (
                "Stream lengths, statuses, and metadata declarations are consistent."
                if not declaration_failures
                else "Published stream declarations contradict data or status lengths."
            ),
        )
    )
    return tuple(findings)
