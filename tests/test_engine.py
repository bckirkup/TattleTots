"""Unit tests for engine components."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.attention import allocate_attention, compute_niche_overlap
from tattletots.engine.compression import (
    AR1Compression,
    PCACompression,
    ThresholdCompression,
    create_compression_model,
)
from tattletots.engine.config import SimulationConfig
from tattletots.engine.reproduction import attempt_reproduction
from tattletots.engine.trophic import compute_trophic_level
from tattletots.engine.trust import verify_reports
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import CompressionType, Genome
from tattletots.models.report import Report
from tattletots.models.user import User


class TestCompression:
    def test_pca_extracts_structure(self) -> None:
        rng = np.random.default_rng(42)
        # Data with clear structure: 2 components in 5D space
        basis = rng.standard_normal((2, 5))
        activations = rng.standard_normal((10, 2))
        data = activations @ basis + rng.standard_normal((10, 5)) * 0.1

        model = PCACompression(n_components=2, efficiency=1.0)
        residual, info_yield = model.fit_transform(data)
        assert info_yield > 0.5  # Should capture most variance

    def test_pca_residual_has_lower_variance(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        model = PCACompression(n_components=2)
        residual, _ = model.fit_transform(data)
        assert np.var(residual) <= np.var(data) + 1e-10

    def test_ar1_learns_autocorrelation(self) -> None:
        model = AR1Compression(n_components=3)
        prev = np.array([1.0, 2.0, 3.0])
        model.fit_transform(prev)  # First step: no prediction
        curr = prev * 0.9 + np.array([0.1, 0.1, 0.1])
        _, info_yield = model.fit_transform(curr)
        assert info_yield > 0  # Should detect autocorrelation

    def test_threshold_detects_anomaly(self) -> None:
        model = ThresholdCompression(n_components=3)
        # Train on normal data
        for _ in range(20):
            model.fit_transform(np.random.randn(5) * 0.1)
        # Inject anomaly
        score = model.anomaly_score(np.ones(5) * 10.0)
        assert score > 2.0  # Well above threshold

    def test_factory_creates_correct_type(self) -> None:
        model = create_compression_model(CompressionType.PCA, 3)
        assert isinstance(model, PCACompression)
        model = create_compression_model(CompressionType.AR1, 3)
        assert isinstance(model, AR1Compression)


class TestTrophic:
    def test_raw_consumer_is_level_1(self) -> None:
        agent_inputs = {"a1": ["stream_raw"]}
        stream_sources = {"stream_raw": None}
        level = compute_trophic_level("a1", agent_inputs, stream_sources)
        assert level == pytest.approx(1.0)

    def test_residual_consumer_is_level_2(self) -> None:
        agent_inputs = {"a1": ["s_raw"], "a2": ["s_residual_a1"]}
        stream_sources = {"s_raw": None, "s_residual_a1": "a1"}
        level = compute_trophic_level("a2", agent_inputs, stream_sources)
        assert level == pytest.approx(2.0)

    def test_chain_depth_3(self) -> None:
        agent_inputs = {
            "a1": ["s_raw"],
            "a2": ["s_res_a1"],
            "a3": ["s_res_a2"],
        }
        stream_sources = {"s_raw": None, "s_res_a1": "a1", "s_res_a2": "a2"}
        level = compute_trophic_level("a3", agent_inputs, stream_sources)
        assert level == pytest.approx(3.0)


class TestAttention:
    def test_attention_sums_to_budget(self) -> None:
        user = User(
            name="test",
            attention_budget=5.0,
            priority_vector=np.array([1.0, 0.0]),
        )
        agents = [
            Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    signal_vector=np.array([1.0, 0.0]),
                )
            ),
            Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    signal_vector=np.array([0.5, 0.5]),
                )
            ),
        ]
        # Set trust
        for a in agents:
            user.trust[a.id] = 0.8

        alloc = allocate_attention(user, agents)
        total = sum(alloc.values())
        assert total == pytest.approx(5.0, rel=1e-5)

    def test_zero_sum_across_agents(self) -> None:
        """Attention is zero-sum: more to A means less to B."""
        user = User(
            name="test",
            attention_budget=1.0,
            priority_vector=np.array([1.0, 0.0]),
        )
        agents = [
            Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    signal_vector=np.array([1.0, 0.0]),
                )
            ),
            Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    signal_vector=np.array([0.1, 0.9]),
                )
            ),
        ]
        user.trust[agents[0].id] = 0.9
        user.trust[agents[1].id] = 0.9

        alloc = allocate_attention(user, agents)
        # Agent 0 is more relevant → gets more attention
        assert alloc[agents[0].id] > alloc[agents[1].id]
        # But total is still exactly the budget
        assert sum(alloc.values()) == pytest.approx(1.0, rel=1e-5)

    def test_niche_overlap(self) -> None:
        a = Agent(state=AgentState(signal_vector=np.array([1.0, 0.0, 0.0])))
        b = Agent(state=AgentState(signal_vector=np.array([1.0, 0.0, 0.0])))
        assert compute_niche_overlap(a, b) == pytest.approx(1.0)

        c = Agent(state=AgentState(signal_vector=np.array([0.0, 1.0, 0.0])))
        assert compute_niche_overlap(a, c) == pytest.approx(0.0)


class TestTrust:
    def test_correct_alarm_builds_trust(self) -> None:
        config = SimulationConfig()
        user = User(name="u1")
        report = Report(
            agent_id="a1",
            target_user_id=user.id,
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
        )
        verified = verify_reports([report], True, {user.id: user}, config)
        assert verified[0].correct is True
        assert user.get_trust("a1") > 0.5

    def test_false_alarm_destroys_trust(self) -> None:
        config = SimulationConfig()
        user = User(name="u1")
        report = Report(
            agent_id="a1",
            target_user_id=user.id,
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
        )
        verify_reports([report], False, {user.id: user}, config)
        assert user.get_trust("a1") < 0.5

    def test_trust_asymmetry(self) -> None:
        """Trust is hard to build, easy to destroy."""
        config = SimulationConfig(trust_delta_pos=0.05, trust_delta_neg=0.2)
        user = User(name="u1")
        # 4 correct alarms
        for _ in range(4):
            r = Report(
                agent_id="a1",
                target_user_id=user.id,
                time_step=1,
                signal_vector=np.array([1.0]),
                confidence=0.9,
                anomaly_score=2.0,
            )
            verify_reports([r], True, {user.id: user}, config)
        trust_after_4_correct = user.get_trust("a1")
        # 1 false alarm
        r = Report(
            agent_id="a1",
            target_user_id=user.id,
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
        )
        verify_reports([r], False, {user.id: user}, config)
        trust_after_false = user.get_trust("a1")
        # One false alarm wipes out multiple correct ones
        assert trust_after_false < trust_after_4_correct - 0.1


class TestReproduction:
    def test_reproduction_below_cap(self) -> None:
        rng = np.random.default_rng(42)
        config = SimulationConfig(max_population=50, mutation_rate=0.1)
        agents = [
            Agent(
                genome=Genome(reproduction_threshold=2.0),
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    energy=EnergyReserves(information=3.0, attention=3.0),
                ),
            )
            for _ in range(5)
        ]
        offspring = attempt_reproduction(agents, config, rng)
        assert len(offspring) > 0
        assert all(o.state.generation == 1 for o in offspring)

    def test_reproduction_respects_cap(self) -> None:
        rng = np.random.default_rng(42)
        config = SimulationConfig(max_population=5, mutation_rate=0.1)
        agents = [
            Agent(
                genome=Genome(reproduction_threshold=1.0),
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    energy=EnergyReserves(information=5.0, attention=5.0),
                ),
            )
            for _ in range(5)
        ]
        offspring = attempt_reproduction(agents, config, rng)
        assert len(offspring) == 0  # Already at cap
