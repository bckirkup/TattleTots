"""Peer observation, whistleblowing suspicion, and witness trust updates.

Agents never read User.trust. They update peer_trust only from observable events:
escalations, dispatch actions, post-dispatch outcomes, and resourcing rewards.
"""

from __future__ import annotations

from tattletots.engine.attention import compute_niche_overlap
from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent, LifecycleStage
from tattletots.models.location import EventLocation
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user import User
from tattletots.models.whistleblower_report import WhistleblowerReport


def _location_distance(a: EventLocation, b: EventLocation) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def can_witness(
    observer: Agent,
    location: EventLocation,
    *,
    min_anomaly: float,
    max_distance: int = 1,
) -> bool:
    if observer.state.lifecycle != LifecycleStage.ADULT or not observer.is_alive:
        return False
    if observer.state.last_anomaly_score < min_anomaly:
        return False
    if observer.state.last_inferred_location is None:
        return False
    return _location_distance(observer.state.last_inferred_location, location) <= max_distance


def record_observable_outcomes(
    agents: dict[str, Agent],
    outcomes: list[ResponseOutcome],
) -> None:
    """Publish dispatch/outcome signals agents can witness about themselves."""
    for outcome in outcomes:
        if not outcome.agent_id:
            continue
        agent = agents.get(outcome.agent_id)
        if agent is None:
            continue
        agent.state.last_observed_dispatch = outcome.dispatched
        if outcome.dispatched:
            agent.state.last_observed_outcome_necessary = outcome.response_necessary
        else:
            agent.state.last_observed_outcome_necessary = None


def observable_prestige(juvenile: Agent, candidate: Agent) -> float:
    """Score a role model from signals any Tot can witness (no User.trust)."""
    reward = candidate.state.last_step_attention_income + candidate.state.last_step_info_subsidy
    dispatch_signal = 0.0
    if candidate.state.last_observed_dispatch:
        if candidate.state.last_observed_outcome_necessary:
            dispatch_signal = 1.0
        elif candidate.state.last_observed_outcome_necessary is False:
            dispatch_signal = -0.5
    peer = juvenile.get_peer_trust(candidate.id)
    return peer + reward + dispatch_signal


def collect_whistleblower_suspicions(
    agents: dict[str, Agent],
    reports: list[Report],
    outcomes: list[ResponseOutcome],
    users: dict[str, User],
    config: SimulationConfig,
) -> list[WhistleblowerReport]:
    """Collect false-report suspicions from observable dispatch outcomes only."""
    suspicions: list[WhistleblowerReport] = []
    living = [a for a in agents.values() if a.is_alive and a.state.lifecycle == LifecycleStage.ADULT]
    user_ids = list(users.keys())
    if not user_ids:
        return suspicions

    outcomes_by_agent = {o.agent_id: o for o in outcomes if o.agent_id}

    for report in reports:
        if not report.verified:
            continue
        outcome = outcomes_by_agent.get(report.agent_id)
        if outcome is None or not outcome.dispatched or outcome.response_necessary:
            continue

        for observer in living:
            if observer.id == report.agent_id:
                continue
            overlap = compute_niche_overlap(observer, agents[report.agent_id])
            spatial = can_witness(
                observer,
                report.location,
                min_anomaly=config.peer_witness_min_anomaly,
            )
            if not spatial and overlap < config.peer_overlap_threshold:
                continue

            score = max(0.5 + report.confidence, overlap)
            if score < config.whistleblower_suspicion_threshold:
                continue

            target_idx = hash(observer.id) % len(user_ids)
            suspicions.append(
                WhistleblowerReport(
                    whistleblower_id=observer.id,
                    accused_agent_id=report.agent_id,
                    target_user_id=user_ids[target_idx],
                    time_step=report.time_step,
                    location=report.location,
                    suspicion_score=score,
                    basis="unnecessary_response",
                )
            )
            observer.update_peer_trust(
                report.agent_id,
                negative=True,
                delta_neg=config.peer_trust_delta_neg,
            )
            observer.state.last_whistleblower_reports_issued += 1

    return suspicions


def apply_peer_witness_trust(
    agents: dict[str, Agent],
    reports: list[Report],
    outcomes: list[ResponseOutcome],
    config: SimulationConfig,
) -> int:
    """Update peer_trust from witnessed escalations, dispatches, outcomes, and rewards."""
    updates = 0
    living = [a for a in agents.values() if a.is_alive and a.state.lifecycle == LifecycleStage.ADULT]
    reports_by_location: dict[EventLocation, list[Report]] = {}
    for report in reports:
        if report.verified:
            reports_by_location.setdefault(report.location, []).append(report)

    for outcome in outcomes:
        if not outcome.dispatched or not outcome.agent_id:
            continue
        subject = agents.get(outcome.agent_id)
        if subject is None:
            continue

        for observer in living:
            if observer.id == outcome.agent_id:
                continue
            if not can_witness(
                observer,
                outcome.location,
                min_anomaly=config.peer_witness_min_anomaly,
            ):
                continue
            overlap = compute_niche_overlap(observer, subject)
            if overlap < config.peer_overlap_threshold:
                continue

            if outcome.response_necessary:
                observer.update_peer_trust(
                    outcome.agent_id, positive=True, delta_pos=config.peer_trust_delta_pos
                )
            else:
                observer.update_peer_trust(
                    outcome.agent_id, negative=True, delta_neg=config.peer_trust_delta_neg
                )
            updates += 1

            if subject.state.last_step_attention_income >= config.peer_witness_reward_threshold:
                observer.update_peer_trust(
                    outcome.agent_id, positive=True, delta_pos=config.peer_trust_delta_pos * 0.5
                )
                updates += 1

        if outcome.response_necessary:
            reporters = {r.agent_id for r in reports_by_location.get(outcome.location, [])}
            for observer in living:
                if observer.id in reporters:
                    continue
                if not can_witness(
                    observer,
                    outcome.location,
                    min_anomaly=config.peer_witness_min_anomaly,
                ):
                    continue
                for reporter_id in reporters:
                    observer.update_peer_trust(
                        reporter_id, positive=True, delta_pos=config.peer_trust_delta_pos * 0.5
                    )
                    updates += 1

    return updates
