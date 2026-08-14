"""Tests for ground-truth-derived spatial null models."""

import pytest

from tattletots.telemetry.spatial_nulls import static_prior_precision


def test_static_prior_holds_report_timing_fixed() -> None:
    precision = static_prior_precision(
        (
            (((0, 0),), 0),
            (((0, 0),), 0),
            (((1, 1),), 3),
        )
    )

    assert precision == pytest.approx(0.0)


def test_static_prior_is_bounded_and_safe_without_reports() -> None:
    assert static_prior_precision(()) == pytest.approx(0.0)
    assert static_prior_precision(((((0, 0),), 0),)) == pytest.approx(0.0)
