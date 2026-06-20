"""Tests for proportional band relevance and priority remapping."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.config import SimulationConfig
from tattletots.engine.relevance import (
    align_user_priorities_to_report_space,
    band_relevance,
    canonical_report_dim,
    remap_priority_bands,
    score_report_relevance,
)
from tattletots.engine.world import World
from tattletots.models.user import User


def _fire_style_priority(n: int, band: str) -> np.ndarray:
    priority = np.zeros(n)
    third = n // 3
    if band == "sector":
        priority[:third] = 1.0
    elif band == "ops":
        priority[third : 2 * third] = 1.0
    else:
        priority[2 * third :] = 1.0
    norm = np.linalg.norm(priority)
    return priority / norm if norm > 0 else priority


class TestBandRelevance:
    def test_same_shape_dot_product(self) -> None:
        priority = np.array([1.0, 0.0, 0.0])
        signal = np.array([0.5, 0.5, 0.5])
        assert band_relevance(priority, signal) == pytest.approx(0.5)

    def test_compressed_signal_hits_ops_middle_band(self) -> None:
        priority = _fire_style_priority(56, "ops")
        signal = np.array([0.0, 2.0])
        sector = _fire_style_priority(56, "sector")
        assert band_relevance(priority, signal) > 0.0
        assert band_relevance(priority, signal) > band_relevance(sector, signal)

    def test_score_report_relevance_is_non_negative(self) -> None:
        user = User(name="Chief", priority_vector=_fire_style_priority(56, "ops"))
        assert score_report_relevance(np.array([-1.0, 2.0]), user) >= 0.0


class TestPriorityRemap:
    def test_remap_preserves_role_mass(self) -> None:
        raw = _fire_style_priority(56, "ops")
        remapped = remap_priority_bands(raw, 30)
        assert remapped.shape == (30,)
        assert np.linalg.norm(remapped) == pytest.approx(1.0)
        mid = remapped[10:20]
        assert float(np.mean(mid)) > float(np.mean(remapped[:5]))

    def test_align_user_priorities_at_setup(self) -> None:
        config = SimulationConfig(initial_population=5, max_steps=5, seed=1)
        world = World(config=config)
        world.add_user(User(name="Chief", priority_vector=_fire_style_priority(56, "ops")))
        world.seed_population()
        before = canonical_report_dim(world)
        report_dim = align_user_priorities_to_report_space(world)
        assert report_dim == before
        chief = next(u for u in world.users.values() if u.name == "Chief")
        assert chief.priority_vector.shape == (report_dim,)
