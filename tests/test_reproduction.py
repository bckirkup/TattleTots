"""Tests for continuous dual-currency reproductive limitation."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.reproduction import attempt_reproduction
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome


def _adult(information: float, attention: float) -> Agent:
    return Agent(
        genome=Genome(reproduction_threshold=2.0),
        state=AgentState(
            lifecycle=LifecycleStage.ADULT,
            energy=EnergyReserves(information=information, attention=attention),
        ),
    )


def test_attention_starved_agent_reproduces_less_than_solvent_agent() -> None:
    starved = _adult(information=4.0, attention=0.2)
    solvent = _adult(information=2.0, attention=2.0)
    config = SimulationConfig(
        initial_population=2,
        max_population=10,
        recombination_probability=0.0,
        reproduction_coupling_strength=1.0,
        seed=42,
    )

    offspring = attempt_reproduction([starved, solvent], config, np.random.default_rng(42))

    assert len(offspring) == 1
    assert offspring[0].state.parent_ids == [solvent.id]
