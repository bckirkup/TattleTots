"""Tests for continuous dual-currency reproductive limitation."""

from __future__ import annotations

import numpy as np
import pytest

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


def _cap_config(*, merit_ordering: bool, population: int, cap: int) -> SimulationConfig:
    return SimulationConfig(
        initial_population=population,
        max_population=cap,
        recombination_probability=0.0,
        reproduction_merit_ordering=merit_ordering,
        seed=42,
    )


def test_binding_cap_rations_by_creation_order_by_default() -> None:
    poorest_first = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(4)]
    config = _cap_config(merit_ordering=False, population=4, cap=6)

    offspring = attempt_reproduction(poorest_first, config, np.random.default_rng(42))

    assert len(offspring) == 2
    parents = [child.state.parent_ids[0] for child in offspring]
    assert parents == [poorest_first[0].id, poorest_first[1].id]


def test_merit_ordering_gives_the_cap_to_the_best_provisioned_parents() -> None:
    poorest_first = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(4)]
    config = _cap_config(merit_ordering=True, population=4, cap=6)

    offspring = attempt_reproduction(poorest_first, config, np.random.default_rng(42))

    assert len(offspring) == 2
    parents = [child.state.parent_ids[0] for child in offspring]
    assert parents == [poorest_first[3].id, poorest_first[2].id]


def test_merit_ordering_does_not_change_which_agents_are_eligible() -> None:
    agents = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(4)]
    unconstrained = _cap_config(merit_ordering=True, population=4, cap=100)

    offspring = attempt_reproduction(agents, unconstrained, np.random.default_rng(42))

    assert {child.state.parent_ids[0] for child in offspring} == {a.id for a in agents}


def test_sufficiency_still_discriminates_between_solvent_agents() -> None:
    lean = _adult(information=2.0, attention=2.0)
    rich = _adult(information=8.0, attention=8.0)

    assert lean.reproduction_limiting_factor() == pytest.approx(1.0)
    assert rich.reproduction_limiting_factor() == pytest.approx(1.0)
    assert rich.reproduction_sufficiency() > lean.reproduction_sufficiency() >= 1.0


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
