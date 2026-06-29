"""Unit tests for engine/whistleblowing.py."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.whistleblowing import (
    compute_dishonesty_score,
    create_output_stream,
    identify_whistleblower_targets,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.stream import Stream, StreamType


class TestIdentifyWhistleblowerTargets:
    def test_returns_output_streams_only(self) -> None:
        streams = {
            "raw1": Stream(id="raw1", stream_type=StreamType.RAW, dimensionality=5),
            "res1": Stream(id="res1", stream_type=StreamType.RESIDUAL, dimensionality=5),
            "out1": Stream(id="out1", stream_type=StreamType.OUTPUT, dimensionality=5),
        }
        agent = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=["raw1", "res1", "out1"],
            )
        )
        targets = identify_whistleblower_targets(agent, streams)
        assert targets == ["out1"]

    def test_returns_empty_when_no_output_streams(self) -> None:
        streams = {
            "raw1": Stream(id="raw1", stream_type=StreamType.RAW, dimensionality=5),
        }
        agent = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=["raw1"],
            )
        )
        targets = identify_whistleblower_targets(agent, streams)
        assert targets == []

    def test_returns_empty_when_no_inputs(self) -> None:
        agent = Agent(state=AgentState(lifecycle=LifecycleStage.ADULT))
        targets = identify_whistleblower_targets(agent, {})
        assert targets == []

    def test_handles_missing_stream_gracefully(self) -> None:
        agent = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=["nonexistent"],
            )
        )
        targets = identify_whistleblower_targets(agent, {})
        assert targets == []


class TestComputeDishonestyScore:
    def test_identical_signals_score_zero(self) -> None:
        signal = np.array([1.0, 2.0, 3.0])
        score = compute_dishonesty_score(signal, signal.copy())
        assert score == pytest.approx(0.0)

    def test_different_signals_positive_score(self) -> None:
        claimed = np.array([1.0, 0.0, 0.0])
        truth = np.array([0.0, 1.0, 0.0])
        score = compute_dishonesty_score(claimed, truth)
        assert score > 0.0

    def test_larger_difference_higher_score(self) -> None:
        truth = np.array([0.0, 0.0, 0.0])
        small_lie = np.array([0.1, 0.0, 0.0])
        big_lie = np.array([10.0, 0.0, 0.0])
        small_score = compute_dishonesty_score(small_lie, truth)
        big_score = compute_dishonesty_score(big_lie, truth)
        assert big_score > small_score

    def test_consensus_averages_with_ground_truth(self) -> None:
        claimed = np.array([5.0, 0.0])
        truth = np.array([0.0, 0.0])
        consensus = np.array([0.0, 0.0])
        score_with = compute_dishonesty_score(claimed, truth, consensus)
        score_without = compute_dishonesty_score(claimed, truth)
        # Both components show dishonesty → averaged score ≈ base score
        assert score_with > 0.0
        assert score_without > 0.0

    def test_mismatched_dimensions_uses_minimum(self) -> None:
        claimed = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.0, 2.0])
        score = compute_dishonesty_score(claimed, truth)
        assert score == pytest.approx(0.0)  # first 2 dims match

    def test_empty_inputs_return_zero(self) -> None:
        assert compute_dishonesty_score(np.array([]), np.array([1.0])) == pytest.approx(0.0)
        assert compute_dishonesty_score(np.array([1.0]), np.array([])) == pytest.approx(0.0)


class TestCreateOutputStream:
    def test_creates_output_stream_with_correct_type(self) -> None:
        agent = Agent(state=AgentState(lifecycle=LifecycleStage.ADULT))
        signal = np.array([1.0, 2.0, 3.0])
        stream = create_output_stream(agent, signal, dim=3)
        assert stream.stream_type == StreamType.OUTPUT
        assert stream.dimensionality == 3
        assert stream.source_agent_id == agent.id
        np.testing.assert_array_equal(stream.current_data, signal)

    def test_label_contains_agent_id_prefix(self) -> None:
        agent = Agent(state=AgentState(lifecycle=LifecycleStage.ADULT))
        stream = create_output_stream(agent, np.array([1.0]), dim=1)
        assert agent.id[:8] in stream.label
