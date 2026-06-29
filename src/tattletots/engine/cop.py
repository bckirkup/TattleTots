"""Common Operating Picture fusion and dispatch target selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tattletots.models.dispatch_target import DispatchTarget

if TYPE_CHECKING:
    from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user import User
from tattletots.models.user_cop import UserCOP, location_key
from tattletots.models.whistleblower_report import WhistleblowerReport


def create_initial_cops(
    users: list[User] | dict[str, User],
    *,
    dispatch_threshold: float = 1.0,
    min_supporting_reports: int = 1,
    min_supporting_weight: float = 0.3,
    decay_factor: float = 0.95,
    responder_dispatch_threshold: float | None = None,
) -> dict[str, UserCOP]:
    """Initialize one COP per user."""
    user_list = users.values() if isinstance(users, dict) else users
    cops: dict[str, UserCOP] = {}
    for user in user_list:
        threshold = dispatch_threshold
        if responder_dispatch_threshold is not None:
            threshold = responder_dispatch_threshold
        cops[user.id] = UserCOP(
            user_id=user.id,
            user_name=user.name,
            dispatch_threshold=threshold,
            min_supporting_reports=min_supporting_reports,
            min_supporting_weight=min_supporting_weight,
            decay_factor=decay_factor,
        )
    return cops


def _report_relevance(
    report: Report,
    user: User,
    adapter: DomainAdapter | None,
) -> float:
    if adapter is not None:
        return max(adapter.score_relevance(report.signal_vector, user), 0.0)
    return max(user.compute_relevance(report.signal_vector), 0.0)


def _fuse_report_into_cop(
    cop: UserCOP,
    report: Report,
    user_id: str,
    user: User,
    *,
    adapter: DomainAdapter | None,
    non_target_weight_scale: float,
) -> None:
    relevance = _report_relevance(report, user, adapter)
    trust = user.get_trust(report.agent_id)
    target_scale = 1.0 if report.target_user_id == user_id else non_target_weight_scale
    weight = trust * relevance * report.confidence * target_scale
    contribution = weight * report.anomaly_score
    belief = cop.get_belief(report.location)
    belief.threat_level += contribution
    belief.supporting_reports += 1
    belief.supporting_weight += weight


def fuse_reports_into_cops(
    cops: dict[str, UserCOP],
    reports: list[Report],
    users: dict[str, User],
    time_step: int,
    *,
    adapter: DomainAdapter | None = None,
    non_target_weight_scale: float = 0.5,
) -> None:
    """Fuse verified escalations into every user's COP with role-specific weights."""
    for cop in cops.values():
        cop.time_step = time_step
        cop.decay()

    for report in reports:
        if not report.verified:
            continue
        for user_id, cop in cops.items():
            user = users.get(user_id)
            if user is None:
                continue
            _fuse_report_into_cop(
                cop,
                report,
                user_id,
                user,
                adapter=adapter,
                non_target_weight_scale=non_target_weight_scale,
            )


def apply_outcomes_to_cops(
    cops: dict[str, UserCOP],
    outcomes: list[ResponseOutcome],
    *,
    reinforce_factor: float = 1.2,
    dampen_factor: float = 0.5,
    confirm_bonus: float = 0.2,
) -> None:
    """Feed post-dispatch judgments back into all user COPs."""
    for outcome in outcomes:
        for cop in cops.values():
            belief = cop.get_belief(outcome.location)
            belief.last_dispatched = outcome.dispatched
            belief.last_response_necessary = outcome.response_necessary
            if not outcome.dispatched:
                continue
            if outcome.response_necessary:
                belief.threat_level = (belief.threat_level + confirm_bonus) * reinforce_factor
            else:
                belief.threat_level *= dampen_factor
                belief.unnecessary_dispatch_count += 1


def select_dispatch_targets(
    cops: dict[str, UserCOP],
    responder_user_id: str,
    reports: list[Report],
) -> list[DispatchTarget]:
    """Select dispatch locations from responder COP (not ground truth)."""
    responder_cop = cops.get(responder_user_id)
    if responder_cop is None:
        return []

    reports_by_location: dict[str, list[Report]] = {}
    for report in reports:
        if not report.verified:
            continue
        key = location_key(report.location)
        reports_by_location.setdefault(key, []).append(report)

    targets: list[DispatchTarget] = []
    for location in responder_cop.locations_above_threshold():
        key = location_key(location)
        belief = responder_cop.get_belief(location)
        targets.append(
            DispatchTarget(
                location=location,
                reports=reports_by_location.get(key, []),
                responder_user_id=responder_user_id,
                cop_threat_level=belief.threat_level,
            )
        )
    targets.sort(key=lambda t: t.cop_threat_level, reverse=True)
    return targets


def apply_whistleblower_to_cops(
    cops: dict[str, UserCOP],
    whistleblower_reports: list[WhistleblowerReport],
    *,
    dampen_factor: float = 0.7,
) -> None:
    """Dampen COP threat at locations flagged by whistleblower suspicion."""
    for wb in whistleblower_reports:
        for cop in cops.values():
            belief = cop.get_belief(wb.location)
            belief.threat_level *= dampen_factor


def cop_telemetry(cops: dict[str, UserCOP], responder_user_id: str) -> dict[str, float | int]:
    """Aggregate COP metrics for output schema."""
    responder = cops.get(responder_user_id)
    if responder is None:
        return {"cop_locations_above_threshold": 0, "cop_max_threat_level": 0.0}
    summary = responder.summary()
    return {
        "cop_locations_above_threshold": int(summary["locations_above_threshold"]),
        "cop_max_threat_level": float(summary["max_threat_level"]),
        "cop_mean_threat_level": float(summary["mean_threat_level"]),
        "cop_belief_count": int(summary["belief_count"]),
    }
