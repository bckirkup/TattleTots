"""Trust dynamics: asymmetric update based on verified outcomes."""

from __future__ import annotations

from tattletots.engine.config import SimulationConfig
from tattletots.models.report import Report
from tattletots.models.user import User


def verify_reports(
    reports: list[Report],
    ground_truth_active: bool,
    users: dict[str, User],
    config: SimulationConfig,
) -> list[Report]:
    """Verify pending reports against ground truth and update user trust.

    Trust is asymmetric: Δ⁻ ≫ Δ⁺ (hard to build, easy to destroy).
    """
    verified: list[Report] = []

    for report in reports:
        report.verified = True
        report.correct = ground_truth_active

        user = users.get(report.target_user_id)
        if user is None:
            verified.append(report)
            continue

        if report.correct:
            user.update_trust(
                report.agent_id,
                correct_alarm=True,
                delta_pos=config.trust_delta_pos,
            )
        else:
            user.update_trust(
                report.agent_id,
                false_alarm=True,
                delta_neg=config.trust_delta_neg,
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
                    missed_event=True,
                    delta_miss=config.trust_delta_miss,
                )
