"""Tests for development and mimesis behaviors."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.development import (
    apply_mimesis,
    juvenile_maintenance_cost,
    select_role_models,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.genome import Genome, MimesisMode, ParentalStrategy


class TestDevelopment:
    def test_juvenile_reduced_maintenance(self) -> None:
        config = SimulationConfig(juvenile_maintenance_fraction=0.5)
        juvenile = Agent(
            genome=Genome(maintenance_cost=0.1),
            state=AgentState(lifecycle=LifecycleStage.JUVENILE),
        )
        adult = Agent(
            genome=Genome(maintenance_cost=0.1),
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        assert juvenile_maintenance_cost(juvenile, config) < juvenile_maintenance_cost(
            adult, config
        )

    def test_mimesis_nudges_preferences(self) -> None:
        config = SimulationConfig(mimesis_learning_rate=0.5)
        juvenile = Agent(
            genome=Genome(
                mimesis_mode=MimesisMode.PARENTAL,
                input_preference=np.array([1.0, 0.0, 0.0]),
            ),
            state=AgentState(
                lifecycle=LifecycleStage.JUVENILE,
                parent_ids=["parent1"],
            ),
        )
        parent = Agent(
            id="parent1",
            genome=Genome(input_preference=np.array([0.0, 1.0, 0.0])),
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )
        apply_mimesis(juvenile, [parent], config)
        assert juvenile.state.input_preference_override.size == 3

    def test_parental_role_model_selection(self) -> None:
        juvenile = Agent(
            genome=Genome(mimesis_mode=MimesisMode.PARENTAL),
            state=AgentState(lifecycle=LifecycleStage.JUVENILE, parent_ids=["p1"]),
        )
        parent = Agent(id="p1", state=AgentState(lifecycle=LifecycleStage.ADULT))
        models = select_role_models(juvenile, {"p1": parent, juvenile.id: juvenile}, {})
        assert len(models) == 1

    def test_live_birth_strategy_exists(self) -> None:
        g = Genome(parental_strategy=ParentalStrategy.LIVE_BIRTH)
        assert g.parental_strategy == ParentalStrategy.LIVE_BIRTH
