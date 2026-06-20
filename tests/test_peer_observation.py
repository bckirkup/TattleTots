"""Tests for observable-only peer trust and whistleblowing."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.peer_observation import (
    apply_peer_witness_trust,
    collect_whistleblower_suspicions,
    observable_prestige,
    record_observable_outcomes,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user import User


def _adult(agent_id: str, *, location: tuple[int, int] = (1, 1), anomaly: float = 0.8) -> Agent:
    return Agent(
        id=agent_id,
        state=AgentState(
            lifecycle=LifecycleStage.ADULT,
            last_inferred_location=location,
            last_anomaly_score=anomaly,
            signal_vector=np.array([1.0, 0.0]),
        ),
    )


class TestObservablePeerTrust:
    def test_witness_updates_peer_trust_from_outcome_not_user_trust(self) -> None:
        config = SimulationConfig(peer_overlap_threshold=0.0)
        reporter = _adult("reporter", location=(2, 2))
        observer = _adult("observer", location=(2, 2))
        agents = {"reporter": reporter, "observer": observer}
        report = Report(
            agent_id="reporter",
            target_user_id="user-1",
            time_step=1,
            signal_vector=np.ones(3),
            confidence=0.9,
            anomaly_score=2.0,
            location=(2, 2),
            verified=True,
            correct=False,
        )
        outcome = ResponseOutcome(
            agent_id="reporter",
            responder_user_id="user-1",
            time_step=1,
            location=(2, 2),
            response_type="suppression",
            dispatched=True,
            problem_severity_before=1.0,
            problem_severity_after=0.0,
            problem_present=True,
            mitigated=True,
            response_necessary=True,
        )
        before = observer.get_peer_trust("reporter")
        apply_peer_witness_trust(agents, [report], [outcome], config)
        assert observer.get_peer_trust("reporter") > before

    def test_unnecessary_dispatch_lowers_peer_trust(self) -> None:
        config = SimulationConfig(peer_overlap_threshold=0.0)
        reporter = _adult("reporter", location=(3, 3))
        observer = _adult("observer", location=(3, 3))
        agents = {"reporter": reporter, "observer": observer}
        report = Report(
            agent_id="reporter",
            target_user_id="user-1",
            time_step=1,
            signal_vector=np.ones(3),
            confidence=0.9,
            anomaly_score=2.0,
            location=(3, 3),
            verified=True,
            correct=True,
        )
        outcome = ResponseOutcome(
            agent_id="reporter",
            responder_user_id="user-1",
            time_step=1,
            location=(3, 3),
            response_type="suppression",
            dispatched=True,
            problem_severity_before=0.0,
            problem_severity_after=0.0,
            problem_present=False,
            mitigated=False,
            response_necessary=False,
        )
        before = observer.get_peer_trust("reporter")
        apply_peer_witness_trust(agents, [report], [outcome], config)
        assert observer.get_peer_trust("reporter") < before

    def test_whistleblower_uses_unnecessary_response_not_ground_truth(self) -> None:
        config = SimulationConfig(
            peer_overlap_threshold=0.0,
            whistleblower_suspicion_threshold=0.4,
        )
        reporter = _adult("reporter", location=(4, 4))
        observer = _adult("observer", location=(4, 4))
        agents = {"reporter": reporter, "observer": observer}
        users = {"user-1": User(name="Chief")}
        report = Report(
            agent_id="reporter",
            target_user_id="user-1",
            time_step=1,
            signal_vector=np.ones(3),
            confidence=0.9,
            anomaly_score=2.0,
            location=(4, 4),
            verified=True,
            correct=True,
        )
        outcome = ResponseOutcome(
            agent_id="reporter",
            responder_user_id="user-1",
            time_step=1,
            location=(4, 4),
            response_type="suppression",
            dispatched=True,
            problem_severity_before=0.0,
            problem_severity_after=0.0,
            problem_present=False,
            mitigated=False,
            response_necessary=False,
        )
        suspicions = collect_whistleblower_suspicions(agents, [report], [outcome], users, config)
        assert len(suspicions) == 1
        assert suspicions[0].basis == "unnecessary_response"

    def test_observable_prestige_ignores_user_trust(self) -> None:
        juvenile = _adult("juvenile")
        candidate = _adult("candidate")
        candidate.state.last_step_attention_income = 0.2
        candidate.state.last_observed_dispatch = True
        candidate.state.last_observed_outcome_necessary = True
        user = User(name="Chief")
        user.trust["candidate"] = 0.0
        score = observable_prestige(juvenile, candidate)
        assert score > juvenile.get_peer_trust("candidate")

    def test_record_observable_outcomes(self) -> None:
        reporter = _adult("reporter")
        agents = {"reporter": reporter}
        outcome = ResponseOutcome(
            agent_id="reporter",
            responder_user_id="user-1",
            time_step=1,
            location=(1, 1),
            response_type="suppression",
            dispatched=True,
            problem_severity_before=1.0,
            problem_severity_after=0.1,
            problem_present=True,
            mitigated=True,
            response_necessary=True,
        )
        record_observable_outcomes(agents, [outcome])
        assert reporter.state.last_observed_dispatch
        assert reporter.state.last_observed_outcome_necessary is True
