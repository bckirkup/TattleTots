"""Trust dynamics: asymmetric update based on verified outcomes."""

from __future__ import annotations

from tattletots.engine.config import SimulationConfig
from tattletots.models.location import EventLocation
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user import TrustOutcome, TrustUpdateDeltas, User
from tattletots.models.whistleblower_report import WhistleblowerReport


def verify_reports(
    reports: list[Report],
    active_locations: frozenset[EventLocation],
    users: dict[str, User],
    config: SimulationConfig,
) -> list[Report]:
    """Verify pending reports against active event locations and update user trust."""
    verified: list[Report] = []

    for report in reports:
        report.verified = True
        report.correct = report.location in active_locations

        user = users.get(report.target_user_id)
        if user is None:
            verified.append(report)
            continue

        if report.correct:
            user.update_trust(
                report.agent_id,
                TrustOutcome.CORRECT_ALARM,
                deltas=TrustUpdateDeltas(pos=config.trust_delta_pos),
            )
        else:
            user.update_trust(
                report.agent_id,
                TrustOutcome.FALSE_ALARM,
                deltas=TrustUpdateDeltas(neg=config.trust_delta_neg),
            )

        verified.append(report)

    return verified


def penalize_missed_events(
    agent_ids_that_missed: list[str],
    users: dict[str, User],
    config: SimulationConfig,
) -> None:
    """Apply trust penalty to agents that failed to escalate a true event."""
    for agent_id in agent_ids_that_missed:
        for user in users.values():
            if user.get_trust(agent_id) > 0:
                user.update_trust(
                    agent_id,
                    TrustOutcome.MISSED_EVENT,
                    deltas=TrustUpdateDeltas(miss=config.trust_delta_miss),
                )


def apply_response_outcome_trust(
    reports: list[Report],
    outcomes: list[ResponseOutcome],
    users: dict[str, User],
    config: SimulationConfig,
) -> int:
    """Update User.trust from post-dispatch responder judgment."""
    reports_by_agent: dict[str, Report] = {r.agent_id: r for r in reports if r.verified}
    updates = 0
    for outcome in outcomes:
        if not outcome.dispatched:
            continue
        report = reports_by_agent.get(outcome.agent_id)
        user_id = report.target_user_id if report else outcome.responder_user_id
        user = users.get(user_id)
        if user is None:
            continue
        if outcome.response_necessary:
            user.update_trust(
                outcome.agent_id,
                TrustOutcome.RESPONSE_NECESSARY,
                deltas=TrustUpdateDeltas(response_necessary=config.trust_delta_response_necessary),
            )
        else:
            user.update_trust(
                outcome.agent_id,
                TrustOutcome.RESPONSE_UNNECESSARY,
                deltas=TrustUpdateDeltas(
                    unnecessary_response=config.trust_delta_unnecessary_response
                ),
            )
        updates += 1
    return updates


def apply_whistleblower_trust(
    whistleblower_reports: list[WhistleblowerReport],
    reports: list[Report],
    outcomes: list[ResponseOutcome],
    users: dict[str, User],
    config: SimulationConfig,
) -> tuple[int, int]:
    """Corroborate or refute whistleblower suspicions."""
    reports_by_agent = {r.agent_id: r for r in reports if r.verified}
    outcomes_by_agent = {o.agent_id: o for o in outcomes}
    corroborated = 0
    refuted = 0

    for wb in whistleblower_reports:
        accused_report = reports_by_agent.get(wb.accused_agent_id)
        outcome = outcomes_by_agent.get(wb.accused_agent_id)
        whistle_user = users.get(wb.target_user_id)
        accused_user = users.get(
            accused_report.target_user_id if accused_report else wb.target_user_id
        )

        confirmed = False
        refuted_flag = False
        if accused_report is not None and accused_report.correct is False:
            confirmed = True
        if outcome is not None and outcome.dispatched and not outcome.response_necessary:
            confirmed = True
        if (
            accused_report is not None
            and accused_report.correct
            and outcome is not None
            and outcome.response_necessary
        ):
            refuted_flag = True

        if confirmed and not refuted_flag:
            corroborated += 1
            if whistle_user:
                whistle_user.update_trust(
                    wb.whistleblower_id,
                    TrustOutcome.WHISTLEBLOWER_CORROBORATED,
                    deltas=TrustUpdateDeltas(
                        whistleblower_corroborated=config.trust_delta_whistleblower_corroborated
                    ),
                )
            if accused_user:
                accused_user.update_trust(
                    wb.accused_agent_id,
                    TrustOutcome.ACCUSED_CORROBORATED,
                    deltas=TrustUpdateDeltas(
                        accused_corroborated=config.trust_delta_accused_corroborated
                    ),
                )
        elif refuted_flag:
            refuted += 1
            if whistle_user:
                whistle_user.update_trust(
                    wb.whistleblower_id,
                    TrustOutcome.WHISTLEBLOWER_REFUTED,
                    deltas=TrustUpdateDeltas(
                        whistleblower_refuted=config.trust_delta_whistleblower_refuted
                    ),
                )

    return corroborated, refuted
