"""Unit tests for core domain models."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import CompressionType, Genome
from tattletots.models.report import Report
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import TrustOutcome, TrustUpdateDeltas, User


class TestEnergyReserves:
    def test_alive_when_both_positive(self) -> None:
        e = EnergyReserves(information=1.0, attention=1.0)
        assert e.is_alive

    def test_dead_when_info_zero(self) -> None:
        e = EnergyReserves(information=0.0, attention=1.0)
        assert not e.is_alive

    def test_dead_when_attn_zero(self) -> None:
        e = EnergyReserves(information=1.0, attention=0.0)
        assert not e.is_alive

    def test_dead_when_negative(self) -> None:
        e = EnergyReserves(information=-0.1, attention=1.0)
        assert not e.is_alive

    def test_total(self) -> None:
        e = EnergyReserves(information=1.5, attention=2.5)
        assert e.total == pytest.approx(4.0)

    def test_apply_deltas(self) -> None:
        e = EnergyReserves(information=1.0, attention=1.0)
        e.apply_info_delta(0.5)
        e.apply_attention_delta(-0.3)
        assert e.information == pytest.approx(1.5)
        assert e.attention == pytest.approx(0.7)


class TestGenome:
    def test_default_genome(self) -> None:
        g = Genome()
        assert g.compression_type == CompressionType.PCA
        assert g.n_components == 3
        assert g.escalation_threshold == pytest.approx(0.7)

    def test_mutation_produces_different_genome(self) -> None:
        rng = np.random.default_rng(42)
        g = Genome(n_components=3, escalation_threshold=0.5)
        mutated = g.mutate(rng, rate=1.0)  # High rate = guaranteed mutations
        # At least one field should differ
        assert (
            mutated.n_components != g.n_components
            or mutated.escalation_threshold != g.escalation_threshold
            or mutated.compression_type != g.compression_type
        )

    def test_recombination_produces_child(self) -> None:
        rng = np.random.default_rng(42)
        parent_a = Genome(compression_type=CompressionType.PCA, n_components=5)
        parent_b = Genome(compression_type=CompressionType.AR1, n_components=2)
        child = Genome.recombine(parent_a, parent_b, rng)
        assert child.compression_type in (CompressionType.PCA, CompressionType.AR1)
        assert child.n_components in (5, 2)

    def test_mutation_respects_bounds(self) -> None:
        rng = np.random.default_rng(0)
        g = Genome(escalation_threshold=0.99)
        for _ in range(100):
            g = g.mutate(rng, rate=1.0)
        assert 0.0 <= g.escalation_threshold <= 1.0
        assert 1 <= g.n_components <= 50


class TestStream:
    def test_create_raw_stream(self) -> None:
        s = Stream(stream_type=StreamType.RAW, dimensionality=10)
        assert s.dimensionality == 10
        assert s.source_agent_id is None

    def test_update_validates_dimensionality(self) -> None:
        s = Stream(stream_type=StreamType.RAW, dimensionality=5)
        with pytest.raises(ValueError, match="dimensionality"):
            s.update(np.ones(3))

    def test_update_accepts_correct_dims(self) -> None:
        s = Stream(stream_type=StreamType.RAW, dimensionality=5)
        data = np.ones(5)
        s.update(data)
        np.testing.assert_array_equal(s.current_data, data)


class TestUser:
    def test_default_trust(self) -> None:
        u = User(name="test")
        assert u.get_trust("unknown_agent") == pytest.approx(0.5)

    def test_trust_correct_alarm(self) -> None:
        u = User(name="test")
        u.update_trust("a1", TrustOutcome.CORRECT_ALARM, deltas=TrustUpdateDeltas(pos=0.1))
        assert u.get_trust("a1") == pytest.approx(0.6)

    def test_trust_false_alarm_asymmetric(self) -> None:
        u = User(name="test")
        u.update_trust("a1", TrustOutcome.CORRECT_ALARM, deltas=TrustUpdateDeltas(pos=0.05))
        u.update_trust("a1", TrustOutcome.FALSE_ALARM, deltas=TrustUpdateDeltas(neg=0.2))
        # Started at 0.5, went to 0.55, then dropped to 0.35
        assert u.get_trust("a1") == pytest.approx(0.35)

    def test_trust_bounded(self) -> None:
        u = User(name="test")
        for _ in range(100):
            u.update_trust("a1", TrustOutcome.CORRECT_ALARM, deltas=TrustUpdateDeltas(pos=0.1))
        assert u.get_trust("a1") <= 1.0

        for _ in range(100):
            u.update_trust("a1", TrustOutcome.FALSE_ALARM, deltas=TrustUpdateDeltas(neg=0.5))
        assert u.get_trust("a1") >= 0.0

    def test_compute_relevance(self) -> None:
        u = User(priority_vector=np.array([1.0, 0.0, 0.0]))
        sig = np.array([0.5, 0.5, 0.5])
        assert u.compute_relevance(sig) == pytest.approx(0.5)

    def test_compute_relevance_compressed_signal_outside_priority_band(self) -> None:
        """Fire Ops Chief-style priority: nonzero only in the middle third."""
        n = 9
        priority = np.zeros(n)
        priority[n // 3 : 2 * n // 3] = 1.0
        priority /= np.linalg.norm(priority)
        user = User(name="Fire Operations Chief", priority_vector=priority)
        compressed = np.array([2.0, 1.5])
        assert user.compute_relevance(compressed) > 0.0


class TestAgent:
    def test_lifecycle_transition(self) -> None:
        g = Genome(development_duration=3)
        a = Agent(genome=g, state=AgentState(lifecycle=LifecycleStage.JUVENILE))
        assert a.state.lifecycle == LifecycleStage.JUVENILE
        a.advance_age()
        a.advance_age()
        assert a.state.lifecycle == LifecycleStage.JUVENILE
        a.advance_age()
        assert a.state.lifecycle == LifecycleStage.ADULT

    def test_can_reproduce_requires_adult_and_energy(self) -> None:
        g = Genome(reproduction_threshold=2.0)
        a = Agent(
            genome=g,
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=1.5, attention=1.5),
            ),
        )
        assert a.can_reproduce

    def test_cannot_reproduce_as_juvenile(self) -> None:
        g = Genome(reproduction_threshold=2.0)
        a = Agent(
            genome=g,
            state=AgentState(
                lifecycle=LifecycleStage.JUVENILE,
                energy=EnergyReserves(information=5.0, attention=5.0),
            ),
        )
        assert not a.can_reproduce

    def test_spawn_offspring(self) -> None:
        rng = np.random.default_rng(42)
        g = Genome(reproduction_threshold=2.0)
        parent = Agent(
            genome=g,
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=3.0, attention=3.0),
            ),
        )
        child = parent.spawn_offspring(rng)
        assert child.id != parent.id
        assert child.state.parent_ids == [parent.id]
        assert child.state.generation == 1
        # Parent paid energy cost
        assert parent.state.energy.information < 3.0

    def test_kill(self) -> None:
        a = Agent()
        a.kill()
        assert not a.is_alive
        assert a.state.lifecycle == LifecycleStage.DEAD


class TestReport:
    def test_create_report(self) -> None:
        r = Report(
            agent_id="a1",
            target_user_id="u1",
            time_step=5,
            signal_vector=np.array([1.0, 2.0]),
            confidence=0.8,
            anomaly_score=1.5,
            location=(1, 2),
        )
        assert not r.verified
        assert r.correct is None
