"""Development, mimesis, and parental investment behaviors."""

from __future__ import annotations

from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent, LifecycleStage
from tattletots.models.genome import MimesisMode, ParentalStrategy
from tattletots.models.user import User


def juvenile_maintenance_cost(agent: Agent, config: SimulationConfig) -> float:
    """Reduced maintenance for juveniles during development."""
    base = agent.genome.maintenance_cost
    if agent.state.lifecycle == LifecycleStage.JUVENILE:
        return base * config.juvenile_maintenance_fraction
    return base


def apply_mimesis(
    juvenile: Agent,
    role_models: list[Agent],
    config: SimulationConfig,
) -> None:
    """Nudge juvenile runtime preferences toward observed role models."""
    if juvenile.state.lifecycle != LifecycleStage.JUVENILE:
        return
    if juvenile.genome.mimesis_mode == MimesisMode.NONE:
        return
    if not role_models:
        return

    lr = config.mimesis_learning_rate
    model = role_models[0]

    # Nudge input preference override
    if model.genome.input_preference.size > 0:
        target = model.state.input_preference_override
        if target.size == 0:
            target = model.genome.input_preference
        if target.size > 0:
            current = juvenile.state.input_preference_override
            if current.size == 0:
                current = juvenile.genome.input_preference.copy()
            if current.size == target.size:
                updated = (1 - lr) * current + lr * target
                total = updated.sum()
                if total > 0:
                    updated /= total
                juvenile.state.input_preference_override = updated

    # Nudge fusion weights
    if model.genome.fusion_weights.size > 0:
        target_fw = model.state.fusion_weights_override
        if target_fw.size == 0:
            target_fw = model.genome.fusion_weights
        if target_fw.size > 0:
            current_fw = juvenile.state.fusion_weights_override
            if current_fw.size == 0:
                current_fw = juvenile.genome.fusion_weights.copy()
            if current_fw.size == target_fw.size:
                updated = (1 - lr) * current_fw + lr * target_fw
                total = updated.sum()
                if total > 0:
                    updated /= total
                juvenile.state.fusion_weights_override = updated


def select_role_models(
    juvenile: Agent,
    agents: dict[str, Agent],
    users: dict[str, User],
) -> list[Agent]:
    """Pick role models for mimesis based on genome mode."""
    mode = juvenile.genome.mimesis_mode
    adults = [
        a
        for a in agents.values()
        if a.is_alive and a.state.lifecycle == LifecycleStage.ADULT and a.id != juvenile.id
    ]
    if not adults:
        return []

    if mode == MimesisMode.PARENTAL:
        parents = [agents[pid] for pid in juvenile.state.parent_ids if pid in agents]
        return [p for p in parents if p.is_alive]

    if mode == MimesisMode.NICHE:
        j_inputs = set(juvenile.state.input_stream_ids)
        return [a for a in adults if j_inputs & set(a.state.input_stream_ids)]

    if mode == MimesisMode.OPPORTUNISTIC:
        best_trust = -1.0
        best: Agent | None = None
        for a in adults:
            trust = max((u.get_trust(a.id) for u in users.values()), default=0.0)
            if trust > best_trust:
                best_trust = trust
                best = a
        return [best] if best is not None else []

    return []


def apply_parental_investment(
    parent: Agent,
    child: Agent,
    config: SimulationConfig,
) -> None:
    """Apply parental strategy effects at reproduction."""
    strategy = parent.genome.parental_strategy
    investment = parent.genome.parental_investment

    if strategy == ParentalStrategy.LIVE_BIRTH:
        subsidy = investment * parent.genome.reproduction_threshold
        parent.state.energy.information -= subsidy / 2
        parent.state.energy.attention -= subsidy / 2
        child.state.energy.information += subsidy / 2
        child.state.energy.attention += subsidy / 2

    # MARSUPIAL curated stream is set up in World._setup_marsupial_stream


def lineage_subsidy_eligible(
    parent: Agent,
    child: Agent,
    config: SimulationConfig,
) -> bool:
    """Check if parent-child lineage signature matches for subsidy."""
    diff = abs(parent.genome.lineage_signature - child.genome.lineage_signature)
    return diff <= config.lineage_signature_tolerance
