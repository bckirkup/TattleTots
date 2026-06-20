"""Tests for per-user Common Operating Picture fusion and dispatch selection."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.cop import (
    apply_outcomes_to_cops,
    create_initial_cops,
    fuse_reports_into_cops,
    select_dispatch_targets,
)
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user import User


def _report(
    agent_id: str,
    user_id: str,
    location: tuple[int, int],
    *,
    anomaly: float = 2.0,
    confidence: float = 0.9,
    correct: bool | None = True,
) -> Report:
    return Report(
        agent_id=agent_id,
        target_user_id=user_id,
        time_step=1,
        signal_vector=np.ones(5),
        confidence=confidence,
        anomaly_score=anomaly,
        location=location,
        verified=True,
        correct=correct,
    )


class TestUserCOPFusion:
    def test_fuse_builds_responder_threat_for_compressed_signal(self) -> None:
        """Responder with middle-band priorities still fuses compressed reports."""
        n = 9
        priority = np.zeros(n)
        priority[n // 3 : 2 * n // 3] = 1.0
        priority /= np.linalg.norm(priority)
        responder = User(name="Ops Chief", priority_vector=priority)
        users = {responder.id: responder}
        cops = create_initial_cops(users, dispatch_threshold=0.5)
        report = Report(
            agent_id="a1",
            target_user_id=responder.id,
            time_step=1,
            signal_vector=np.array([2.0, 1.5]),
            confidence=0.9,
            anomaly_score=1.0,
            location=(4, 5),
            verified=True,
            correct=True,
        )
        fuse_reports_into_cops(cops, [report], users, 1, adapter=None)
        belief = cops[responder.id].get_belief((4, 5))
        assert belief.threat_level > 0
        assert belief.supporting_weight > 0

    def test_fuse_builds_threat_at_report_location(self) -> None:
        user = User(name="Responder", priority_vector=np.ones(5))
        users = {user.id: user}
        cops = create_initial_cops(users, dispatch_threshold=0.5)
        report = _report("a1", user.id, (3, 4))
        fuse_reports_into_cops(cops, [report], users, 1, adapter=None)
        belief = cops[user.id].get_belief((3, 4))
        assert belief.threat_level > 0
        assert belief.supporting_reports == 1

    def test_select_dispatch_targets_uses_responder_cop_not_ground_truth(self) -> None:
        responder = User(name="Ops Chief", priority_vector=np.ones(5))
        other = User(name="Other", priority_vector=np.ones(5))
        users = {responder.id: responder, other.id: other}
        cops = create_initial_cops(users, dispatch_threshold=0.1)
        false_report = _report("a1", other.id, (1, 1), correct=False)
        fuse_reports_into_cops(cops, [false_report], users, 1, adapter=None)

        targets = select_dispatch_targets(cops, responder.id, [false_report])
        assert len(targets) == 1
        assert targets[0].location == (1, 1)

    def test_outcome_feedback_dampens_unnecessary_location(self) -> None:
        user = User(name="Responder", priority_vector=np.ones(5))
        users = {user.id: user}
        cops = create_initial_cops(users, dispatch_threshold=0.1)
        report = _report("a1", user.id, (2, 2))
        fuse_reports_into_cops(cops, [report], users, 1, adapter=None)
        before = cops[user.id].get_belief((2, 2)).threat_level

        outcome = ResponseOutcome(
            agent_id="a1",
            responder_user_id=user.id,
            time_step=1,
            location=(2, 2),
            response_type="test",
            dispatched=True,
            problem_severity_before=0.0,
            problem_severity_after=0.0,
            problem_present=False,
            mitigated=False,
            response_necessary=False,
        )
        apply_outcomes_to_cops(cops, [outcome])
        after = cops[user.id].get_belief((2, 2)).threat_level
        assert after < before


class TestTrustFromOutcomes:
    def test_response_outcome_trust_updates_user(self) -> None:
        from tattletots.engine.trust import apply_response_outcome_trust

        user = User(name="Chief")
        users = {user.id: user}
        config = SimulationConfig()
        report = _report("agent-1", user.id, (0, 0))
        outcome = ResponseOutcome(
            agent_id="agent-1",
            responder_user_id=user.id,
            time_step=1,
            location=(0, 0),
            response_type="suppression",
            dispatched=True,
            problem_severity_before=1.0,
            problem_severity_after=0.2,
            problem_present=True,
            mitigated=True,
            response_necessary=True,
        )
        initial = user.get_trust("agent-1")
        apply_response_outcome_trust([report], [outcome], users, config)
        assert user.get_trust("agent-1") > initial
