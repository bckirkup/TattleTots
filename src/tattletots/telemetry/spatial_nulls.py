"""Spatial null models derived from ground-truth event windows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from tattletots.models.location import EventLocation


def static_prior_precision(
    windows: Iterable[tuple[Sequence[EventLocation], int]],
) -> float:
    """Measure a modal-location reporter with report timing held fixed.

    Ground-truth locations vary across the measured windows, while each window's
    report count is held fixed. The reporter always names the modal location and
    receives one correctness opportunity for each report issued in a window.
    """
    measured_windows = list(windows)
    report_count = sum(max(reports, 0) for _, reports in measured_windows)
    if report_count == 0:
        return 0.0
    location_counts = Counter(
        location for locations, _ in measured_windows for location in locations
    )
    if not location_counts:
        return 0.0
    prior_location = location_counts.most_common(1)[0][0]
    correct_reports = sum(
        max(reports, 0) for locations, reports in measured_windows if prior_location in locations
    )
    return correct_reports / report_count
