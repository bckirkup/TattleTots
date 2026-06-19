"""Tests for temporal memory fusion."""

from __future__ import annotations

import numpy as np

from tattletots.engine.temporal import apply_temporal_fusion
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, TemporalFusionMode


class TestTemporal:
    def test_none_mode_passthrough(self) -> None:
        agent = Agent(
            genome=Genome(temporal_memory_depth=0, temporal_fusion_mode=TemporalFusionMode.NONE)
        )
        data = np.ones(5)
        out = apply_temporal_fusion(agent, data)
        np.testing.assert_array_equal(out, data)

    def test_ema_blends_history(self) -> None:
        agent = Agent(
            genome=Genome(temporal_memory_depth=5, temporal_fusion_mode=TemporalFusionMode.EMA)
        )
        apply_temporal_fusion(agent, np.zeros(4))
        out = apply_temporal_fusion(agent, np.ones(4))
        assert out.size == 4
        assert 0.0 < float(out.mean()) < 1.0

    def test_window_stack_averages(self) -> None:
        agent = Agent(
            genome=Genome(
                temporal_memory_depth=3, temporal_fusion_mode=TemporalFusionMode.WINDOW_STACK
            )
        )
        apply_temporal_fusion(agent, np.zeros(3))
        apply_temporal_fusion(agent, np.ones(3) * 2)
        out = apply_temporal_fusion(agent, np.ones(3) * 4)
        assert out.size == 3

    def test_deeper_memory_increases_compute_cost(self) -> None:
        from tattletots.engine.config import SimulationConfig

        shallow = Genome(temporal_memory_depth=0)
        deep = Genome(temporal_memory_depth=50)
        config = SimulationConfig()
        assert deep.total_compute_cost(config) > shallow.total_compute_cost(config)
