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
