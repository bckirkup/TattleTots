"""Domestication (niche construction): downstream agents shape upstream compression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.models.agent import Agent


def compute_shaping_signal(
    downstream_agent: Agent,
    _upstream_agent: Agent,
    useful_dimensions: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute a feedback signal from downstream to upstream agent.

    The shaping signal hints at what kind of residual would be most useful
    to the downstream consumer. This is niche construction: higher-level
    agents shape their food sources.

    Only effective when signal components overlap (per requirements §6.5).
    """
    if useful_dimensions is None:
        # Default: signal what the downstream agent is currently extracting
        return downstream_agent.state.signal_vector

    return useful_dimensions


def apply_shaping(
    upstream_agent: Agent,
    shaping_signals: list[NDArray[np.float64]],
) -> None:
    """Apply accumulated shaping signals to modify upstream agent's compression preferences.

    The domestication_sensitivity genome parameter controls how much
    the agent responds to downstream pressure.
    """
    if not shaping_signals or upstream_agent.genome.domestication_sensitivity <= 0:
        return

    sensitivity = upstream_agent.genome.domestication_sensitivity

    # Average the shaping signals
    valid_signals = [s for s in shaping_signals if s.size > 0]
    if not valid_signals:
        return

    # Pad to same length
    max_len = max(len(s) for s in valid_signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in valid_signals]
    mean_signal = np.mean(padded, axis=0)

    # Read current effective preference (state override > genome default)
    pref = upstream_agent.state.input_preference_override
    if pref.size == 0:
        pref = upstream_agent.genome.input_preference
    if pref.size == 0 or pref.size != len(mean_signal):
        return

    # Nudge preferences toward the shaping signal direction
    new_pref = pref + sensitivity * mean_signal[: len(pref)]
    new_pref = np.clip(new_pref, 0.0, None)
    total = new_pref.sum()
    if total > 0:
        new_pref /= total
    upstream_agent.state.input_preference_override = new_pref
