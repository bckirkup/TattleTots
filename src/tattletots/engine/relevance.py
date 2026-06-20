"""Role-weighted relevance between user priorities and compressed agent signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tattletots.models.user import User

if TYPE_CHECKING:
    from tattletots.engine.world import World


def remap_priority_bands(priority: np.ndarray, target_dim: int) -> np.ndarray:
    """Resample a raw-stream priority vector into compressed report space."""
    n_p = len(priority)
    if target_dim <= 0 or n_p == 0:
        return np.array([], dtype=np.float64)
    if target_dim == n_p:
        return priority.astype(np.float64, copy=True)

    out = np.zeros(target_dim, dtype=np.float64)
    for ti in range(target_dim):
        p_lo = int(ti * n_p / target_dim)
        p_hi = int((ti + 1) * n_p / target_dim)
        if p_hi <= p_lo:
            p_hi = min(p_lo + 1, n_p)
        out[ti] = float(np.mean(priority[p_lo:p_hi]))

    norm = float(np.linalg.norm(out))
    if norm > 1e-12:
        out /= norm
    return out


def band_relevance(priority: np.ndarray, signal: np.ndarray) -> float:
    """Dot-product relevance with proportional band mapping across dimensions.

    User priorities are defined in raw stream space (e.g. fire role bands).
    Agent reports carry compressed signals of varying length. Each signal
    component is weighted by the mean priority mass in the corresponding
    proportional band of the priority vector.
    """
    n_p = len(priority)
    n_s = len(signal)
    if n_p == 0 or n_s == 0:
        return 0.0
    if n_p == n_s:
        return float(np.dot(priority, signal))

    total = 0.0
    for si in range(n_s):
        p_lo = int(si * n_p / n_s)
        p_hi = int((si + 1) * n_p / n_s)
        if p_hi <= p_lo:
            p_hi = min(p_lo + 1, n_p)
        band = priority[p_lo:p_hi]
        band_weight = float(np.mean(band)) if band.size else 0.0
        total += band_weight * float(signal[si])
    return total


def score_report_relevance(signal_vector: np.ndarray, user: User) -> float:
    """Default relevance: role-weighted band alignment (non-negative)."""
    return max(band_relevance(user.priority_vector, signal_vector), 0.0)


def canonical_report_dim(world: World) -> int:
    """Typical compressed signal width used for priority remapping at setup."""
    dims = [
        min(
            agent.genome.working_dim,
            world.config.max_working_dim,
            world.config.max_stream_dim,
        )
        for agent in world.agents.values()
    ]
    if dims:
        return int(np.median(dims))
    return min(world.config.default_working_dim, world.config.max_stream_dim)


def align_user_priorities_to_report_space(world: World) -> int:
    """Remap user priorities from raw stream dims to median agent report dims."""
    report_dim = canonical_report_dim(world)
    for user in world.users.values():
        if user.priority_vector.size > 0 and user.priority_vector.size != report_dim:
            user.priority_vector = remap_priority_bands(user.priority_vector, report_dim)
    return report_dim
