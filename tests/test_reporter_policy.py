"""Tests for the first-class reporter-policy seam."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.interface.reporter_policy import (
    ReporterDecision,
    ReporterPolicyContext,
    register_reporter_policy,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.genome import Genome
from tattletots.models.observation import ObservationStatus, StreamMetadata
from tattletots.models.report import Report
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User


class _EvidencePolicy:
    contexts: list[ReporterPolicyContext] = []

    def decide(self, context: ReporterPolicyContext) -> ReporterDecision:
        self.contexts.append(context)
        return ReporterDecision(escalate=True, location=(99, -4))


def _register_policy() -> None:
    _EvidencePolicy.contexts = []
    register_reporter_policy("test-evidence-policy", _EvidencePolicy)


def _world(*, require_grounded: bool = False) -> tuple[World, Agent]:
    world = World(
        config=SimulationConfig(
            initial_population=2,
            max_population=10,
            seed=7,
            require_grounded_report_locations=require_grounded,
        )
    )
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        label="published",
        metadata=StreamMetadata(
            coordinates=[(1.0, 2.0), None],
            sensor_coordinates=[None, (3.0, 4.0)],
        ),
        current_status=np.array(
            [ObservationStatus.OBSERVED.value, ObservationStatus.MISSING.value]
        ),
        current_data=np.array([1.0, 0.0]),
    )
    world.add_stream(stream)
    world.add_user(User(name="user"))
    agent = Agent(
        genome=Genome(
            reporter_policy="test-evidence-policy",
            escalation_threshold=0.0,
            working_dim=8,
        ),
        state=AgentState(
            lifecycle=LifecycleStage.ADULT,
            input_stream_ids=[stream.id],
        ),
    )
    world.agents[agent.id] = agent
    world._init_agent_model(agent)
    return world, agent


def test_unknown_policy_fails_during_agent_initialization() -> None:
    world = World(config=SimulationConfig(initial_population=2, seed=3))
    agent = Agent(genome=Genome(reporter_policy="does-not-exist"))
    with pytest.raises(ValueError, match="unknown reporter policy"):
        world._init_agent_model(agent)


def test_policy_context_contains_only_read_only_published_evidence() -> None:
    _register_policy()
    world, agent = _world()

    world._compress(agent)
    report = world._maybe_escalate(agent, raw_anomaly=1.0)

    assert report is not None
    context = _EvidencePolicy.contexts[-1]
    assert {field.name for field in fields(context)} == {
        "observation",
        "signal_vector",
        "anomaly_score",
        "escalation_threshold",
        "time_step",
        "location_frame",
        "streams",
    }
    assert not any(
        name in {field.name for field in fields(context)}
        for name in ("active_locations", "world", "adapter", "agents")
    )
    assert not context.observation.flags.writeable
    assert not context.signal_vector.flags.writeable
    assert len(context.streams) == 1
    assert context.streams[0].label == "published"
    assert not context.streams[0].data.flags.writeable
    assert context.streams[0].observation_status == (
        ObservationStatus.OBSERVED.value,
        ObservationStatus.MISSING.value,
    )
    assert context.streams[0].metadata.coordinates == ((1.0, 2.0), None)
    assert context.streams[0].metadata.sensor_coordinates == (None, (3.0, 4.0))


def test_policy_location_is_projected_and_passes_grounded_gate() -> None:
    _register_policy()
    world, agent = _world(require_grounded=True)
    world.set_location_frame(((0, 0), (5, 5)))

    world._compress(agent)
    report = world._maybe_escalate(agent, raw_anomaly=1.0)

    assert report is not None
    assert report.location == (5, 0)
    assert agent.state.last_geometry_location == (1, 2)
    assert agent.state.policy_report_location == (5, 0)


def test_escalating_policy_must_name_a_location() -> None:
    class _MissingLocationPolicy:
        def decide(self, _context: ReporterPolicyContext) -> ReporterDecision:
            return ReporterDecision(escalate=True)

    register_reporter_policy("missing-location-policy", _MissingLocationPolicy)
    world, agent = _world()
    agent.genome.reporter_policy = "missing-location-policy"
    world._init_agent_model(agent)

    world._compress(agent)
    with pytest.raises(ValueError, match="escalated without a location"):
        world._maybe_escalate(agent, raw_anomaly=1.0)


def test_reporter_policy_tag_is_inherited_not_mutated_or_randomized() -> None:
    rng = np.random.default_rng(11)
    parent = Genome(reporter_policy="test-evidence-policy")
    child = parent.mutate(rng, rate=1.0)
    assert child.reporter_policy == parent.reporter_policy

    ordinary = Genome()
    assert ordinary.mutate(rng, rate=1.0).reporter_policy is None
    assert Genome.random_genome(rng).reporter_policy is None

    values = {
        Genome.recombine(
            parent,
            ordinary,
            np.random.default_rng(seed),
        ).reporter_policy
        for seed in range(12)
    }
    assert values <= {"test-evidence-policy", None}
    assert values


def test_reporter_group_telemetry_is_separate_from_step_fingerprint_payload() -> None:
    _register_policy()
    world, agent = _world()
    world.set_event_state([(5, 0)])
    world.set_location_frame(((0, 0), (5, 5)))
    record = world.step()

    assert record.reports_issued == 1
    groups = world.telemetry.reporter_group_history[-1]
    assert groups["designed_population_share"] == pytest.approx(1.0)
    assert groups["designed_reports"] == 1
    assert groups["ordinary_reports"] == 0
    assert groups["designed_correct_reports"] == 1
    assert groups["ordinary_correct_reports"] == 0


def test_reporter_group_uses_dead_authors_and_preserves_report_totals() -> None:
    world, agent = _world()
    user_id = next(iter(world.users))
    report = Report(
        agent_id=agent.id,
        target_user_id=user_id,
        time_step=0,
        signal_vector=np.zeros(1),
        confidence=1.0,
        anomaly_score=1.0,
        location=(1, 2),
        verified=True,
        correct=True,
    )
    unknown_report = Report(
        agent_id="author-no-longer-present",
        target_user_id=user_id,
        time_step=0,
        signal_vector=np.zeros(1),
        confidence=1.0,
        anomaly_score=1.0,
        location=(1, 2),
    )
    agent.kill()

    record = world._build_step_record(
        reports=[report, unknown_report],
        births=[],
        deaths=[agent.id],
        missed=[],
    )
    groups = world._last_reporter_groups

    assert record.reports_issued == 2
    assert groups["designed_reports"] == 1
    assert groups["designed_correct_reports"] == 1
    assert groups["ordinary_reports"] == 1
    assert groups["ordinary_correct_reports"] == 0
    assert groups["designed_reports"] + groups["ordinary_reports"] == record.reports_issued
