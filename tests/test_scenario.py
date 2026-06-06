"""Unit tests for scenarios/gaussian_shift.py."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


class TestGaussianShiftScenario:
    def test_streams_have_correct_total_dimensionality(self) -> None:
        scenario = GaussianShiftScenario(dimensionality=20, seed=42)
        streams = scenario.get_streams()
        total_dim = sum(s.dimensionality for s in streams)
        assert total_dim == 20

    def test_streams_are_raw_type(self) -> None:
        scenario = GaussianShiftScenario(seed=42)
        for stream in scenario.get_streams():
            assert stream.stream_type.value == "raw"

    def test_users_have_orthogonal_priorities(self) -> None:
        scenario = GaussianShiftScenario(n_components=10, seed=42)
        users = scenario.get_users()
        assert len(users) == 2
        dot = float(np.dot(users[0].priority_vector, users[1].priority_vector))
        assert dot == pytest.approx(0.0, abs=1e-10)

    def test_step_updates_stream_data(self) -> None:
        scenario = GaussianShiftScenario(seed=42)
        streams = scenario.get_streams()
        # Before step, streams have zero data
        initial_data = [s.current_data.copy() for s in streams]
        scenario.step(0)
        # After step, at least one stream should have non-zero data
        any_changed = any(
            not np.array_equal(s.current_data, init)
            for s, init in zip(streams, initial_data, strict=True)
        )
        assert any_changed

    def test_data_changes_after_shift_step(self) -> None:
        scenario = GaussianShiftScenario(shift_step=50, seed=42)
        # Collect data before shift
        pre_data = []
        for step in range(45, 50):
            scenario.step(step)
            pre_data.append(np.concatenate([s.current_data for s in scenario.get_streams()]))
        # Collect data after shift
        post_data = []
        for step in range(50, 55):
            scenario.step(step)
            post_data.append(np.concatenate([s.current_data for s in scenario.get_streams()]))
        # The structure changed — data should differ (in distribution, not just noise)
        pre_mean = np.mean(pre_data, axis=0)
        post_mean = np.mean(post_data, axis=0)
        # Means won't be identical (structural change)
        assert not np.allclose(pre_mean, post_mean, atol=0.1)

    def test_ground_truth_window_around_shift(self) -> None:
        scenario = GaussianShiftScenario(shift_step=100)
        # Far from shift → False
        assert not scenario.get_ground_truth(0)
        assert not scenario.get_ground_truth(50)
        assert not scenario.get_ground_truth(200)
        # Within 5 steps of shift → True
        assert scenario.get_ground_truth(95)
        assert scenario.get_ground_truth(100)
        assert scenario.get_ground_truth(105)
        # Just outside → False
        assert not scenario.get_ground_truth(94)
        assert not scenario.get_ground_truth(106)

    def test_compute_costs_returns_expected_keys(self) -> None:
        scenario = GaussianShiftScenario(seed=42)
        costs = scenario.compute_costs(n_escalations=10, n_correct=7, n_false_alarms=3, n_missed=2)
        assert set(costs.keys()) == {"surveillance_cost", "response_cost", "damage_cost"}
        assert costs["surveillance_cost"] > 0
        assert costs["response_cost"] > 0
        assert costs["damage_cost"] > 0

    def test_to_config_and_from_config_roundtrip(self) -> None:
        original = GaussianShiftScenario(
            n_components=8, dimensionality=16, noise_std=0.3, shift_step=150, total_steps=300
        )
        config = original.to_config()
        restored = GaussianShiftScenario.from_config(config)
        assert restored.n_components == 8
        assert restored.dimensionality == 16
        assert restored.noise_std == 0.3
        assert restored.shift_step == 150
        assert restored.total_steps == 300

    def test_from_config_uses_defaults(self) -> None:
        scenario = GaussianShiftScenario.from_config({})
        assert scenario.n_components == 10
        assert scenario.dimensionality == 20
        assert scenario.noise_std == 0.5

    def test_deterministic_with_same_seed(self) -> None:
        s1 = GaussianShiftScenario(seed=123)
        s2 = GaussianShiftScenario(seed=123)
        s1.step(0)
        s2.step(0)
        for stream1, stream2 in zip(s1.get_streams(), s2.get_streams(), strict=True):
            np.testing.assert_array_equal(stream1.current_data, stream2.current_data)

    def test_score_relevance_delegates_to_user(self) -> None:
        scenario = GaussianShiftScenario(seed=42)
        users = scenario.get_users()
        signal = np.array([1.0] * 10)
        score = scenario.score_relevance(signal, users[0])
        assert isinstance(score, float)
