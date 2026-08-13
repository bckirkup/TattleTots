"""Unit tests for engine components."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tattletots.engine.attention import allocate_attention, compute_niche_overlap
from tattletots.engine.compression import (
    AR1Compression,
    CompressionModel,
    PCACompression,
    ThresholdCompression,
    WaveletCompression,
    create_compression_model,
)
from tattletots.engine.config import SimulationConfig
from tattletots.engine.domestication import apply_shaping, compute_shaping_signal
from tattletots.engine.reproduction import attempt_reproduction
from tattletots.engine.trophic import compute_trophic_level
from tattletots.engine.trust import verify_reports
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import CompressionType, Genome
from tattletots.models.report import Report
from tattletots.models.user import User


class TestCompression:
    @pytest.mark.parametrize(
        ("factory", "change"),
        [
            (lambda: PCACompression(n_components=2), np.array([5.0, -5.0, 5.0, -5.0])),
            (lambda: AR1Compression(n_components=2), np.array([5.0, -5.0, 5.0, -5.0])),
            (
                lambda: ThresholdCompression(n_components=2),
                np.array([5.0, -5.0, 5.0, -5.0]),
            ),
            (
                lambda: WaveletCompression(n_components=2),
                np.array([5.0, -5.0, 5.0, -5.0]),
            ),
        ],
    )
    def test_every_compressor_scores_pre_update_distribution_change(
        self,
        factory: Callable[[], CompressionModel],
        change: np.ndarray,
    ) -> None:
        model = factory()
        baseline = np.array([0.1, -0.1, 0.05, -0.05])
        for _ in range(12):
            model.fit_transform(baseline)

        score_before = model.anomaly_score(change)
        _residual, _yield, observed_score = model.observe(change)

        assert score_before > 1e-6
        assert observed_score == pytest.approx(score_before)

    @pytest.mark.parametrize("compression_type", list(CompressionType))
    def test_observe_contract_scores_before_update_generically(
        self,
        compression_type: CompressionType,
    ) -> None:
        baseline = np.array([0.1, -0.1, 0.05, -0.05])
        change = np.array([5.0, -5.0, 5.0, -5.0])
        model = create_compression_model(compression_type, n_components=2)
        fresh_model = create_compression_model(compression_type, n_components=2)
        for _ in range(12):
            model.fit_transform(baseline)
            fresh_model.fit_transform(baseline)

        _residual, _yield, observed_score = model.observe(change)
        assert observed_score == pytest.approx(fresh_model.anomaly_score(change))

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: PCACompression(n_components=2),
            lambda: AR1Compression(n_components=2),
            lambda: ThresholdCompression(n_components=2),
            lambda: WaveletCompression(n_components=2),
        ],
    )
    def test_compressor_dimension_change_resets_pre_update_state(
        self,
        factory: Callable[[], CompressionModel],
    ) -> None:
        model = factory()
        model.fit_transform(np.zeros(4))
        _residual, _yield, score = model.observe(np.ones(5))

        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize("compression_type", list(CompressionType))
    def test_yield_is_invariant_to_multiplicative_scale(
        self,
        compression_type: CompressionType,
    ) -> None:
        baseline = np.array([-0.7, 0.2, 1.1, 2.4, -1.3, 0.8, 3.2, -2.1])
        current = np.array([0.4, -1.2, 2.1, 0.3, 1.7, -0.5, 2.8, -1.4])
        yields: list[float] = []

        for scale in (1e-6, 1.0, 1e6):
            model = create_compression_model(compression_type, n_components=3)
            model.fit_transform(baseline * scale)
            _residual, info_yield, _score = model.observe(current * scale)
            yields.append(info_yield)

        assert yields[1] == pytest.approx(yields[0], abs=1e-10)
        assert yields[2] == pytest.approx(yields[0], abs=1e-10)

    @pytest.mark.parametrize("compression_type", list(CompressionType))
    def test_yield_is_invariant_to_constant_offset(
        self,
        compression_type: CompressionType,
    ) -> None:
        baseline = np.array([-0.7, 0.2, 1.1, 2.4, -1.3, 0.8, 3.2, -2.1])
        current = np.array([0.4, -1.2, 2.1, 0.3, 1.7, -0.5, 2.8, -1.4])
        yields: list[float] = []

        for offset in (-1e6, 0.0, 1e6):
            model = create_compression_model(compression_type, n_components=3)
            model.fit_transform(baseline + offset)
            _residual, info_yield, _score = model.observe(current + offset)
            yields.append(info_yield)

        assert yields[1] == pytest.approx(yields[0], abs=1e-10)
        assert yields[2] == pytest.approx(yields[0], abs=1e-10)

    @pytest.mark.parametrize("compression_type", list(CompressionType))
    def test_dimension_reset_cannot_reopen_scale_dependent_yield(
        self,
        compression_type: CompressionType,
    ) -> None:
        yields: list[float] = []
        reset_input = np.array([0.4, -1.2, 2.1, 0.3, 1.7])

        for scale in (1e-6, 1.0, 1e6):
            model = create_compression_model(compression_type, n_components=3)
            model.fit_transform(np.ones(4) * scale)
            _residual, info_yield, _score = model.observe(reset_input * scale)
            yields.append(info_yield)

        assert yields[1] == pytest.approx(yields[0], abs=1e-10)
        assert yields[2] == pytest.approx(yields[0], abs=1e-10)

    def test_pca_extracts_structure(self) -> None:
        rng = np.random.default_rng(42)
        # Data with clear structure: 2 components in 5D space
        basis = rng.standard_normal((2, 5))
        activations = rng.standard_normal((10, 2))
        data = activations @ basis + rng.standard_normal((10, 5)) * 0.1

        model = PCACompression(n_components=2, efficiency=1.0)
        yields = [model.fit_transform(row)[1] for row in data]
        info_yield = yields[-1]
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
        rng = np.random.default_rng(0)
        for _ in range(20):
            model.fit_transform(rng.standard_normal(5) * 0.1)
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
            location=(2, 3),
        )
        verified = verify_reports([report], frozenset({(2, 3)}), {user.id: user}, config)
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
            location=(0, 0),
        )
        verify_reports([report], frozenset(), {user.id: user}, config)
        assert user.get_trust("a1") < 0.5

    def test_wrong_location_is_false_even_when_event_active(self) -> None:
        config = SimulationConfig()
        user = User(name="u1")
        report = Report(
            agent_id="a1",
            target_user_id=user.id,
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
            location=(1, 1),
        )
        verified = verify_reports([report], frozenset({(5, 5)}), {user.id: user}, config)
        assert verified[0].correct is False

    def test_trust_asymmetry(self) -> None:
        """Trust is hard to build, easy to destroy."""
        config = SimulationConfig(trust_delta_pos=0.05, trust_delta_neg=0.2)
        user = User(name="u1")
        active = frozenset({(0, 0)})
        # 4 correct alarms
        for _ in range(4):
            r = Report(
                agent_id="a1",
                target_user_id=user.id,
                time_step=1,
                signal_vector=np.array([1.0]),
                confidence=0.9,
                anomaly_score=2.0,
                location=(0, 0),
            )
            verify_reports([r], active, {user.id: user}, config)
        trust_after_4_correct = user.get_trust("a1")
        # 1 false alarm
        r = Report(
            agent_id="a1",
            target_user_id=user.id,
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
            location=(9, 9),
        )
        verify_reports([r], active, {user.id: user}, config)
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


class TestDomestication:
    """Domestication unit tests (moved from test_math_properties.py)."""

    def test_domestication_with_overlapping_signals(self) -> None:
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )
        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])
        assert upstream.state.input_preference_override.size > 0
        override = upstream.state.input_preference_override
        assert override[0] > 0.5

    def test_domestication_without_overlap(self) -> None:
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.2]),
            ),
        )
        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])
        assert upstream.state.input_preference_override.size == 0

    def test_domestication_zero_sensitivity(self) -> None:
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.0,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )
        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])
        assert upstream.state.input_preference_override.size == 0

    def test_domestication_does_not_mutate_genome(self) -> None:
        original_pref = np.array([0.5, 0.3, 0.2])
        genome = Genome(
            input_preference=original_pref.copy(),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )
        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])
        np.testing.assert_array_almost_equal(
            upstream.genome.input_preference,
            original_pref,
        )
