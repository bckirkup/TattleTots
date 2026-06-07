"""Unit tests for engine/world.py internals.

Tests the private methods of World in isolation by constructing minimal
state rather than running full simulations.
"""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.compression import PCACompression, ThresholdCompression
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import CompressionType, Genome
from tattletots.models.report import Report
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User


def _minimal_world(seed: int = 42, **config_kw: object) -> World:
    """Build a World with no agents/streams for controlled testing."""
    config = SimulationConfig(
        initial_population=2,
        max_population=50,
        seed=seed,
        **config_kw,  # type: ignore[arg-type]
    )
    return World(config=config)


def _add_raw_stream(world: World, dim: int = 5, data: np.ndarray | None = None) -> Stream:
    """Add a raw stream and return it."""
    s = Stream(stream_type=StreamType.RAW, dimensionality=dim, label="test_raw")
    if data is not None:
        s.current_data = data
    else:
        s.current_data = np.ones(dim)
    world.add_stream(s)
    return s


class TestSeedPopulation:
    def test_creates_correct_number_of_agents(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        world.seed_population()
        assert len(world.agents) == 2

    def test_all_agents_start_alive(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        world.seed_population()
        for agent in world.agents.values():
            assert agent.is_alive

    def test_all_agents_get_compression_models(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        world.seed_population()
        for agent_id in world.agents:
            assert agent_id in world.compression_models

    def test_all_agents_get_residual_streams(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        world.seed_population()
        for agent in world.agents.values():
            assert agent.state.output_stream_id is not None
            assert agent.state.output_stream_id in world.streams

    def test_custom_genomes(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        genomes = [
            Genome(compression_type=CompressionType.PCA, n_components=2),
            Genome(compression_type=CompressionType.AR1, n_components=3),
        ]
        world.seed_population(genomes=genomes)
        types = {a.genome.compression_type for a in world.agents.values()}
        assert CompressionType.PCA in types
        assert CompressionType.AR1 in types


class TestInitAgentModel:
    def test_creates_correct_compression_model(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        agent = Agent(
            genome=Genome(compression_type=CompressionType.THRESHOLD, n_components=3),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=list(world.streams.keys()),
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        assert isinstance(world.compression_models[agent.id], ThresholdCompression)

    def test_creates_residual_stream(self) -> None:
        world = _minimal_world()
        s = _add_raw_stream(world, dim=7)
        agent = Agent(
            genome=Genome(compression_type=CompressionType.PCA),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        out_id = agent.state.output_stream_id
        assert out_id is not None
        assert world.streams[out_id].stream_type == StreamType.RESIDUAL
        assert world.streams[out_id].source_agent_id == agent.id


class TestCompress:
    def test_updates_signal_vector(self) -> None:
        world = _minimal_world()
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10)
        s = _add_raw_stream(world, dim=10, data=data)
        agent = Agent(
            genome=Genome(compression_type=CompressionType.THRESHOLD, n_components=3),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        world._compress(agent)
        assert agent.state.signal_vector.size > 0

    def test_caps_input_at_max_stream_dim(self) -> None:
        world = _minimal_world()
        # Two streams with 20 dims each = 40 total, should be capped to max_stream_dim (30)
        s1 = _add_raw_stream(world, dim=20, data=np.ones(20))
        s2 = Stream(
            stream_type=StreamType.RAW,
            dimensionality=20,
            label="test_raw_2",
            current_data=np.ones(20),
        )
        world.add_stream(s2)
        agent = Agent(
            genome=Genome(compression_type=CompressionType.THRESHOLD, n_components=3),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s1.id, s2.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        world._compress(agent)
        # The residual stream should be <= max_stream_dim
        out_stream = world.streams[agent.state.output_stream_id]
        assert out_stream.dimensionality <= world.config.max_stream_dim

    def test_custom_max_stream_dim(self) -> None:
        world = _minimal_world(max_stream_dim=15)
        s1 = _add_raw_stream(world, dim=20, data=np.ones(20))
        agent = Agent(
            genome=Genome(compression_type=CompressionType.THRESHOLD, n_components=3),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s1.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        world._compress(agent)
        out_stream = world.streams[agent.state.output_stream_id]
        assert out_stream.dimensionality <= 15

    def test_no_input_yields_zero(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        agent = Agent(
            genome=Genome(compression_type=CompressionType.PCA),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[],  # no inputs
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        world._compress(agent)
        assert agent.state.last_step_yield == 0.0


class TestMaybeEscalate:
    def test_no_escalation_below_threshold(self) -> None:
        world = _minimal_world()
        data = np.zeros(5)  # zero data → zero anomaly
        s = _add_raw_stream(world, dim=5, data=data)
        user = User(name="u1")
        world.add_user(user)
        agent = Agent(
            genome=Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=0.9,
            ),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        # Train on zero data — build baseline history
        for _ in range(5):
            world._compress(agent)
            world._maybe_escalate(agent)  # accumulate anomaly history
        report = world._maybe_escalate(agent)
        assert report is None

    def test_no_escalation_with_insufficient_history(self) -> None:
        """With < 3 anomaly samples, normalized score is 0.0 → no escalation."""
        world = _minimal_world()
        s = _add_raw_stream(world, dim=5, data=np.ones(5) * 100)
        user = User(name="u1")
        world.add_user(user)
        agent = Agent(
            genome=Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=0.01,
            ),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        world._compress(agent)
        # First call → only 1 sample in history → anomaly = 0.0
        report = world._maybe_escalate(agent)
        assert report is None
        assert len(agent.state.anomaly_history) == 1

    def test_escalation_on_anomaly(self) -> None:
        world = _minimal_world()
        s = _add_raw_stream(world, dim=5, data=np.zeros(5))
        user = User(name="u1")
        world.add_user(user)
        agent = Agent(
            genome=Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=0.01,  # very low threshold
            ),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        # Build baseline on zero data
        for _ in range(10):
            world._compress(agent)
            world._maybe_escalate(agent)  # accumulate anomaly history
        # Inject a large anomaly — z-score will be very high
        s.current_data = np.ones(5) * 100.0
        world._compress(agent)
        report = world._maybe_escalate(agent)
        assert report is not None
        assert report.agent_id == agent.id

    def test_no_report_without_users(self) -> None:
        world = _minimal_world()
        s = _add_raw_stream(world, dim=5, data=np.zeros(5))
        agent = Agent(
            genome=Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=0.01,
            ),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        # Build baseline on zero data (no users registered)
        for _ in range(5):
            world._compress(agent)
            world._maybe_escalate(agent)
        # Inject anomaly
        s.current_data = np.ones(5) * 100.0
        world._compress(agent)
        report = world._maybe_escalate(agent)
        # No users to report to → None even though anomaly is high
        assert report is None

    def test_anomaly_history_window_capped(self) -> None:
        """Anomaly history should not grow beyond the window size."""
        world = _minimal_world()
        s = _add_raw_stream(world, dim=5, data=np.ones(5))
        user = User(name="u1")
        world.add_user(user)
        agent = Agent(
            genome=Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=0.99,
            ),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                input_stream_ids=[s.id],
            ),
        )
        world.agents[agent.id] = agent
        world._init_agent_model(agent)
        for _ in range(80):
            world._compress(agent)
            world._maybe_escalate(agent)
        assert len(agent.state.anomaly_history) <= world._ANOMALY_WINDOW


class TestApplyEnergy:
    def test_compute_cost_reduces_info_energy(self) -> None:
        world = _minimal_world()
        agent = Agent(
            genome=Genome(compute_cost=0.1, maintenance_cost=0.01),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=1.0, attention=1.0),
                last_step_yield=0.0,
            ),
        )
        world._apply_energy(agent, {}, [])
        assert agent.state.energy.information == pytest.approx(0.9)

    def test_yield_offsets_cost(self) -> None:
        world = _minimal_world()
        agent = Agent(
            genome=Genome(compute_cost=0.1, maintenance_cost=0.01),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=1.0, attention=1.0),
                last_step_yield=0.2,
            ),
        )
        world.compression_models[agent.id] = PCACompression(n_components=2)
        world._apply_energy(agent, {}, [])
        # info_delta = -0.1 + 0.2 = +0.1
        assert agent.state.energy.information == pytest.approx(1.1)

    def test_false_alarm_penalty_reduces_attention(self) -> None:
        world = _minimal_world(false_alarm_penalty=0.5)
        agent = Agent(
            genome=Genome(compute_cost=0.01, maintenance_cost=0.01),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=1.0, attention=1.0),
            ),
        )
        false_report = Report(
            agent_id=agent.id,
            target_user_id="u1",
            time_step=1,
            signal_vector=np.array([1.0]),
            confidence=0.9,
            anomaly_score=2.0,
            verified=True,
            correct=False,
        )
        world._apply_energy(agent, {}, [false_report])
        assert agent.state.energy.attention < 1.0

    def test_subsidy_from_downstream_consumers(self) -> None:
        world = _minimal_world(subsidy_rate=0.1)
        # Upstream agent with a residual stream
        upstream = Agent(
            genome=Genome(compute_cost=0.01, maintenance_cost=0.01),
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=1.0, attention=1.0),
            ),
        )
        residual = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=5,
            source_agent_id=upstream.id,
        )
        world.add_stream(residual)
        upstream.state.output_stream_id = residual.id
        world.agents[upstream.id] = upstream

        # Two downstream agents consuming the residual
        for _ in range(2):
            downstream = Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    input_stream_ids=[residual.id],
                ),
            )
            world.agents[downstream.id] = downstream

        world._apply_energy(upstream, {}, [])
        # subsidy = 2 * 0.1 = 0.2; info_delta = -0.01 + 0.2 = 0.19
        assert upstream.state.energy.information == pytest.approx(1.19)


class TestApplyDomestication:
    def test_domestication_flows_downstream_to_upstream(self) -> None:
        world = _minimal_world()
        upstream = Agent(
            genome=Genome(
                input_preference=np.array([0.5, 0.3, 0.2]),
                domestication_sensitivity=0.5,
            ),
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        residual = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=5,
            source_agent_id=upstream.id,
        )
        world.add_stream(residual)
        upstream.state.output_stream_id = residual.id
        world.agents[upstream.id] = upstream

        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
                input_stream_ids=[residual.id],
            ),
        )
        world.agents[downstream.id] = downstream

        world._apply_domestication([upstream, downstream])
        assert upstream.state.input_preference_override.size > 0


class TestBuildStepRecord:
    def test_record_captures_correct_counts(self) -> None:
        world = _minimal_world()
        # Add 3 living agents
        for _ in range(3):
            a = Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    energy=EnergyReserves(information=1.0, attention=1.0),
                ),
            )
            world.agents[a.id] = a
        # 1 dead agent
        dead = Agent(state=AgentState(lifecycle=LifecycleStage.DEAD))
        world.agents[dead.id] = dead

        world.time_step = 5
        record = world._build_step_record(
            reports=[],
            births=["b1", "b2"],
            deaths=["d1"],
            missed=[],
        )
        assert record.time_step == 5
        assert record.population == 3
        assert record.births == 2
        assert record.deaths == 1

    def test_record_handles_no_living_agents(self) -> None:
        world = _minimal_world()
        record = world._build_step_record(reports=[], births=[], deaths=[], missed=[])
        assert record.population == 0
        assert record.mean_info_energy == 0.0
        assert record.mean_attn_energy == 0.0

    def test_missed_events_propagated(self) -> None:
        world = _minimal_world()
        record = world._build_step_record(
            reports=[],
            births=[],
            deaths=[],
            missed=["a1", "a2", "a3"],
        )
        assert record.missed_events == 3


class TestWorldStep:
    def test_single_step_produces_telemetry(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world, dim=5)
        user = User(name="test_user", attention_budget=1.0)
        world.add_user(user)
        world.seed_population()
        record = world.step()
        assert record.time_step == 1
        assert record.population >= 0
        assert world.telemetry.total_steps == 1

    def test_dead_agents_streams_cleaned_up(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world, dim=5)
        user = User(name="test_user", attention_budget=1.0)
        world.add_user(user)
        world.seed_population()

        # Kill all agents
        for agent in world.agents.values():
            agent.state.energy.information = -1.0
            agent.state.energy.attention = -1.0

        world.step()

        # Residual streams should be cleaned up
        residual_streams = [
            s for s in world.streams.values() if s.stream_type == StreamType.RESIDUAL
        ]
        assert len(residual_streams) == 0

    def test_run_produces_records(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world, dim=5)
        user = User(name="test_user", attention_budget=1.0)
        world.add_user(user)
        world.seed_population()
        records = world.run(steps=5)
        assert len(records) == 5


class TestRandomGenome:
    def test_genome_fields_in_valid_ranges(self) -> None:
        world = _minimal_world()
        _add_raw_stream(world)
        user = User(name="test")
        world.add_user(user)
        for _ in range(20):
            g = world._random_genome()
            assert isinstance(g.compression_type, CompressionType)
            assert 1 <= g.n_components <= 5
            assert 0.3 <= g.escalation_threshold <= 0.9
            assert 0.5 <= g.metabolic_efficiency <= 2.0
            assert 0.05 <= g.compute_cost <= 0.2
            assert 0.02 <= g.maintenance_cost <= 0.1
            assert 1.5 <= g.reproduction_threshold <= 3.0
            assert 0.0 <= g.domestication_sensitivity <= 0.3
            assert g.input_preference.sum() == pytest.approx(1.0, abs=1e-10)
            assert g.target_user_affinity.sum() == pytest.approx(1.0, abs=1e-10)
