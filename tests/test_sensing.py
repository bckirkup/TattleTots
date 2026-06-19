"""Tests for sensing and fusion pipeline."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.sensing import prepare_agent_input
from tattletots.models.agent import Agent, AgentState
from tattletots.models.genome import Genome, SensingStrategy
from tattletots.models.stream import Stream, StreamType


def _agent_with_streams(strategy: SensingStrategy, working_dim: int = 16) -> tuple[Agent, dict[str, Stream]]:
    s1 = Stream(stream_type=StreamType.RAW, dimensionality=10, label="s1", current_data=np.arange(10.0))
    s2 = Stream(stream_type=StreamType.RAW, dimensionality=10, label="s2", current_data=np.arange(10.0) * 2)
    streams = {s1.id: s1, s2.id: s2}
    agent = Agent(
        genome=Genome(
            sensing_strategy=strategy,
            working_dim=working_dim,
            fusion_weights=np.array([0.6, 0.4]),
            block_index=1,
            dim_offset=7,
        ),
        state=AgentState(input_stream_ids=[s1.id, s2.id]),
    )
    return agent, streams


class TestSensing:
    def test_concat_reduces_to_working_dim(self) -> None:
        agent, streams = _agent_with_streams(SensingStrategy.CONCAT, working_dim=12)
        out, _ = prepare_agent_input(agent, streams, SimulationConfig())
        assert out.size == 12

    def test_weighted_fuse_produces_working_dim(self) -> None:
        agent, streams = _agent_with_streams(SensingStrategy.WEIGHTED_FUSE, working_dim=8)
        out, _ = prepare_agent_input(agent, streams, SimulationConfig())
        assert out.size == 8

    def test_subspace_sample_stable(self) -> None:
        agent, streams = _agent_with_streams(SensingStrategy.SUBSPACE_SAMPLE, working_dim=8)
        out1, _ = prepare_agent_input(agent, streams, SimulationConfig())
        out2, _ = prepare_agent_input(agent, streams, SimulationConfig())
        np.testing.assert_array_equal(out1, out2)

    def test_block_specialize_selects_block(self) -> None:
        agent, streams = _agent_with_streams(SensingStrategy.BLOCK_SPECIALIZE, working_dim=10)
        config = SimulationConfig(n_spatial_blocks=5)
        out, _ = prepare_agent_input(agent, streams, config)
        assert out.size == 10

    def test_projection_does_not_increase_variance_unbounded(self) -> None:
        agent, streams = _agent_with_streams(SensingStrategy.CONCAT, working_dim=8)
        combined = np.concatenate([streams[sid].current_data for sid in agent.state.input_stream_ids])
        out, _ = prepare_agent_input(agent, streams, SimulationConfig())
        assert float(np.var(out)) <= float(np.var(combined)) + 1e-6
