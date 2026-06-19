"""Escalation decision logic with adaptive threshold calibration."""

from __future__ import annotations

import numpy as np

from tattletots.engine.compression import CompressionModel
from tattletots.models.agent import Agent
from tattletots.models.genome import EscalationMode


def normalize_anomaly(
    agent: Agent,
    raw_anomaly: float,
) -> float:
    """Map raw anomaly to normalized score using rolling baseline."""
    depth = agent.genome.escalation_memory_depth
    history = agent.state.anomaly_history
    history.append(raw_anomaly)
    if len(history) > depth:
        agent.state.anomaly_history = history[-depth:]
        history = agent.state.anomaly_history

    if len(history) < 3:
        return 0.0

    hist = np.array(history[:-1])
    mu = float(np.mean(hist))
    sigma = max(float(np.std(hist)), 1e-10)
    z = (raw_anomaly - mu) / sigma
    z_clipped = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-0.5 * (z_clipped - 2.0))))


def compute_effective_threshold(agent: Agent) -> float:
    """Compute escalation threshold based on genome mode."""
    genome = agent.genome
    base = genome.escalation_threshold

    if genome.escalation_mode == EscalationMode.FIXED:
        return base

    history = agent.state.anomaly_history
    if len(history) < 3:
        return base

    hist = np.array(history)

    if genome.escalation_mode == EscalationMode.ADAPTIVE_QUANTILE:
        # escalation_threshold encodes target quantile (0.85-0.99 typical)
        q = float(np.clip(base, 0.5, 0.99))
        return float(np.quantile(hist, q))

    if genome.escalation_mode == EscalationMode.ADAPTIVE_VOLATILITY:
        rolling_std = float(np.std(hist))
        return float(np.clip(base + genome.threshold_adaptation_rate * rolling_std, 0.0, 1.0))

    return base


def should_escalate(
    agent: Agent,
    model: CompressionModel,
    combined_input: np.ndarray,
) -> tuple[float, float, bool]:
    """Score anomaly and decide escalation.

    Returns (normalized_anomaly, effective_threshold, should_fire).
    """
    raw = model.anomaly_score(combined_input)
    anomaly = normalize_anomaly(agent, raw)
    threshold = compute_effective_threshold(agent)
    agent.state.effective_escalation_threshold = threshold
    return anomaly, threshold, anomaly >= threshold
