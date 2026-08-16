"""Tests for adaptive escalation."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.compression import ThresholdCompression
from tattletots.engine.escalation import (
    compute_effective_threshold,
    normalize_anomaly,
    should_escalate,
)
from tattletots.models.agent import Agent
from tattletots.models.genome import EscalationMode, Genome


def _spiky_anomalies(count: int, seed: int) -> list[float]:
    """A raw-anomaly stream with a quiet baseline and occasional spikes.

    A flat stream normalizes to a single repeated score, which no quantile can
    discriminate; real anomaly streams are heavy-tailed, so the fixture is too.
    """
    rng = np.random.default_rng(seed)
    baseline = rng.gamma(shape=2.0, scale=0.5, size=count)
    spikes = rng.random(count) < 0.2
    return [float(value + 12.0 * spike) for value, spike in zip(baseline, spikes, strict=True)]


class TestEscalation:
    def test_fixed_threshold_unchanged(self) -> None:
        agent = Agent(genome=Genome(escalation_mode=EscalationMode.FIXED, escalation_threshold=0.7))
        assert compute_effective_threshold(agent) == pytest.approx(0.7)

    def test_insufficient_history_returns_zero_anomaly(self) -> None:
        agent = Agent()
        score = normalize_anomaly(agent, 1.0)
        assert score == pytest.approx(0.0)

    def test_adaptive_quantile_uses_history(self) -> None:
        agent = Agent(
            genome=Genome(
                escalation_mode=EscalationMode.ADAPTIVE_QUANTILE,
                escalation_threshold=0.9,
            )
        )
        agent.state.anomaly_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold = compute_effective_threshold(agent)
        assert 0.0 <= threshold <= 1.0

    def test_score_unit_calibration_tracks_the_compared_distribution(self) -> None:
        """Quantile thresholds land inside the score range they are compared against."""
        thresholds: list[float] = []
        for quantile in (0.5, 0.75, 0.9, 0.99):
            agent = Agent(
                genome=Genome(
                    escalation_mode=EscalationMode.ADAPTIVE_QUANTILE,
                    escalation_threshold=quantile,
                    escalation_memory_depth=50,
                )
            )
            for raw in _spiky_anomalies(40, seed=7):
                normalize_anomaly(agent, raw)
            threshold = compute_effective_threshold(agent, score_units=True)
            scores = agent.state.normalized_anomaly_history
            assert min(scores) <= threshold <= max(scores)
            thresholds.append(threshold)
        assert thresholds == sorted(thresholds)
        assert thresholds[-1] > thresholds[0]

    def test_raw_unit_calibration_can_leave_the_comparable_range(self) -> None:
        """Raw-window calibration returns a compression-scale number, not a 0-1 score."""
        agent = Agent(
            genome=Genome(
                escalation_mode=EscalationMode.ADAPTIVE_QUANTILE,
                escalation_threshold=0.9,
                escalation_memory_depth=50,
            )
        )
        agent.state.anomaly_history = [float(value) for value in range(1, 40)]
        agent.state.normalized_anomaly_history = [0.2, 0.25, 0.3, 0.35]
        raw_units = compute_effective_threshold(agent, score_units=False)
        score_units = compute_effective_threshold(agent, score_units=True)
        assert raw_units > 1.0
        assert 0.0 <= score_units <= 1.0
        assert score_units < raw_units

    def test_escalation_share_rises_as_the_target_quantile_falls(self) -> None:
        """Calibrating in score units makes the quantile trait control firing rate."""
        shares: list[float] = []
        model = ThresholdCompression(n_components=2)
        data = np.zeros(5)
        for quantile in (0.95, 0.8, 0.6):
            agent = Agent(
                genome=Genome(
                    escalation_mode=EscalationMode.ADAPTIVE_QUANTILE,
                    escalation_threshold=quantile,
                    escalation_memory_depth=40,
                )
            )
            raws = _spiky_anomalies(120, seed=11)
            fires = 0
            for raw in raws:
                _, _, fire = should_escalate(agent, model, data, raw_anomaly=raw, score_units=True)
                fires += int(fire)
            shares.append(fires / len(raws))
        assert shares == sorted(shares)
        assert shares[-1] - shares[0] > 0.05
        assert all(0.0 <= share <= 1.0 for share in shares)

    def test_fixed_mode_ignores_the_calibration_units(self) -> None:
        """Negative control: the knob only touches the adaptive modes."""
        agent = Agent(genome=Genome(escalation_mode=EscalationMode.FIXED, escalation_threshold=0.7))
        agent.state.anomaly_history = [10.0, 20.0, 30.0]
        agent.state.normalized_anomaly_history = [0.1, 0.2, 0.3]
        assert compute_effective_threshold(agent, score_units=True) == pytest.approx(0.7)
        assert compute_effective_threshold(agent, score_units=False) == pytest.approx(0.7)

    def test_normalized_history_is_bounded_and_capped(self) -> None:
        agent = Agent(genome=Genome(escalation_memory_depth=10))
        rng = np.random.default_rng(3)
        for _ in range(60):
            normalize_anomaly(agent, float(rng.normal(0.0, 5.0)))
        history = agent.state.normalized_anomaly_history
        assert len(history) == 10
        assert all(np.isfinite(score) for score in history)
        assert min(history) >= 0.0
        assert max(history) <= 1.0

    def test_should_escalate_fires_on_high_anomaly(self) -> None:
        agent = Agent(genome=Genome(escalation_threshold=0.01))
        model = ThresholdCompression(n_components=2)
        data = np.zeros(5)
        for _ in range(10):
            model.fit_transform(data)
            should_escalate(agent, model, data)
        data = np.ones(5) * 100
        model.fit_transform(data)
        anomaly, threshold, fire = should_escalate(agent, model, data)
        assert fire or anomaly >= threshold
