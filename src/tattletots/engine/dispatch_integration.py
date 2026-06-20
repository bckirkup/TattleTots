"""Shared COP-driven dispatch and post-dispatch feedback loop for domain integrations."""

from __future__ import annotations

from tattletots.engine.config import SimulationConfig
from tattletots.engine.cop import (
    apply_outcomes_to_cops,
    create_initial_cops,
    fuse_reports_into_cops,
    select_dispatch_targets,
)
from tattletots.engine.response_judgment import patch_step_record_responses
from tattletots.engine.world import World
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.user_cop import UserCOP
from tattletots.models.whistleblower_report import WhistleblowerReport
from tattletots.telemetry.recorder import StepRecord


def init_user_cops(
    world: World,
    adapter: DomainAdapter,
    config: SimulationConfig,
) -> dict[str, UserCOP]:
    """Create per-user COPs with responder-specific dispatch threshold."""
    responder_id = adapter.get_responder_user_id()
    cops = create_initial_cops(
        world.users,
        dispatch_threshold=config.cop_dispatch_threshold,
        min_supporting_reports=config.cop_min_supporting_reports,
        min_supporting_weight=config.cop_min_supporting_weight,
        decay_factor=config.cop_decay_factor,
    )
    if responder_id in cops:
        cops[responder_id].dispatch_threshold = config.cop_dispatch_threshold
    return cops


def run_dispatch_cycle(
    world: World,
    adapter: DomainAdapter,
    cops: dict[str, UserCOP],
    time_step: int,
    config: SimulationConfig,
) -> tuple[list[ResponseOutcome], list[WhistleblowerReport]]:
    """Fuse reports into COPs, dispatch from responder belief, apply feedback."""
    fuse_reports_into_cops(
        cops,
        world.last_reports,
        world.users,
        time_step,
        adapter=adapter,
        non_target_weight_scale=config.cop_non_target_weight_scale,
    )

    targets = select_dispatch_targets(
        cops,
        adapter.get_responder_user_id(),
        world.last_reports,
    )
    outcomes = adapter.dispatch_and_judge_responses(targets, time_step)

    apply_outcomes_to_cops(
        cops,
        outcomes,
        reinforce_factor=config.cop_reinforce_factor,
        dampen_factor=config.cop_dampen_factor,
        confirm_bonus=config.cop_confirm_bonus,
    )

    whistleblower_reports = world.apply_post_dispatch_feedback(outcomes, cops=cops)
    patch_step_record_responses(world.telemetry.history, outcomes)
    return outcomes, whistleblower_reports


def patch_step_record_cop(
    history: list[StepRecord],
    cops: dict[str, UserCOP],
    responder_user_id: str,
) -> None:
    """Attach COP summary metrics to the latest step record."""
    if not history:
        return
    from tattletots.engine.cop import cop_telemetry

    record = history[-1]
    for key, value in cop_telemetry(cops, responder_user_id).items():
        if hasattr(record, key):
            setattr(record, key, value)
