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


def _cap_config(
    *,
    merit_ordering: bool,
    population: int,
    cap: int,
    correctness_weight: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        initial_population=population,
        max_population=cap,
        recombination_probability=0.0,
        reproduction_merit_ordering=merit_ordering,
        reproduction_correctness_weight=correctness_weight,
        seed=42,
    )


def _reporter(*, information: float, attention: float, reports: int, correct: int) -> Agent:
    agent = _adult(information=information, attention=attention)
    agent.state.reports_issued = reports
    agent.state.correct_reports = correct
    return agent


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


def _accurate_last() -> list[Agent]:
    """Four eligible parents whose reserve order is the reverse of their correctness."""
    return [
        _reporter(information=8.0, attention=8.0, reports=8, correct=0),
        _reporter(information=6.0, attention=6.0, reports=8, correct=2),
        _reporter(information=4.0, attention=4.0, reports=8, correct=5),
        _reporter(information=2.0, attention=2.0, reports=8, correct=8),
    ]


def test_correctness_weight_defaults_to_the_reserves_only_ordering() -> None:
    """At the default weight a binding cap still goes to the best-provisioned parents."""
    parents = _accurate_last()
    config = _cap_config(merit_ordering=True, population=4, cap=6)

    offspring = attempt_reproduction(parents, config, np.random.default_rng(42))

    assert [child.state.parent_ids[0] for child in offspring] == [parents[0].id, parents[1].id]


def test_full_correctness_weight_gives_the_cap_to_the_most_accurate_parents() -> None:
    parents = _accurate_last()
    config = _cap_config(merit_ordering=True, population=4, cap=6, correctness_weight=1.0)

    offspring = attempt_reproduction(parents, config, np.random.default_rng(42))

    assert [child.state.parent_ids[0] for child in offspring] == [parents[3].id, parents[2].id]


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_reproductive_share_of_accurate_parents_is_graded_in_the_weight(weight: float) -> None:
    """Correctness rank displaces reserve rank progressively, not all at once."""
    parents = _accurate_last()
    config = _cap_config(merit_ordering=True, population=4, cap=6, correctness_weight=weight)

    offspring = attempt_reproduction(parents, config, np.random.default_rng(42))
    accurate_ids = {parents[2].id, parents[3].id}
    accurate_share = sum(
        1 for child in offspring if child.state.parent_ids[0] in accurate_ids
    ) / len(offspring)

    assert 0.0 <= accurate_share <= 1.0
    if weight <= 0.25:
        assert accurate_share == pytest.approx(0.0)
    if weight >= 0.75:
        assert accurate_share == pytest.approx(1.0)


def test_one_lucky_report_does_not_outrank_a_sustained_reporter() -> None:
    """Shrinkage keeps a 1/1 record below a longer record of the same accuracy."""
    lucky = _reporter(information=2.0, attention=2.0, reports=1, correct=1)
    sustained = _reporter(information=2.0, attention=2.0, reports=8, correct=8)
    config = _cap_config(merit_ordering=True, population=2, cap=3, correctness_weight=1.0)

    offspring = attempt_reproduction([lucky, sustained], config, np.random.default_rng(42))

    assert len(offspring) == 1
    assert offspring[0].state.parent_ids == [sustained.id]


def test_silent_parents_rank_below_any_correct_reporter() -> None:
    silent = _reporter(information=8.0, attention=8.0, reports=0, correct=0)
    correct = _reporter(information=2.0, attention=2.0, reports=4, correct=3)
    config = _cap_config(merit_ordering=True, population=2, cap=3, correctness_weight=1.0)

    offspring = attempt_reproduction([silent, correct], config, np.random.default_rng(42))

    assert len(offspring) == 1
    assert offspring[0].state.parent_ids == [correct.id]


def test_correctness_weight_does_not_change_which_agents_are_eligible() -> None:
    parents = _accurate_last()
    unconstrained = _cap_config(merit_ordering=True, population=4, cap=100, correctness_weight=1.0)

    offspring = attempt_reproduction(parents, unconstrained, np.random.default_rng(42))

    assert {child.state.parent_ids[0] for child in offspring} == {a.id for a in parents}


def _share_config(share: float, *, correctness_weight: float = 1.0) -> SimulationConfig:
    return SimulationConfig(
        initial_population=8,
        max_population=100,
        recombination_probability=0.0,
        reproduction_merit_ordering=True,
        reproduction_correctness_weight=correctness_weight,
        reproduction_recruitment_share=share,
        seed=42,
    )


@pytest.mark.parametrize(
    ("share", "expected"),
    [(1.0, 8), (0.5, 4), (0.25, 2), (0.1, 1)],
)
def test_recruitment_share_grades_how_many_eligible_parents_reproduce(
    share: float, expected: int
) -> None:
    """Fewer eligible parents recruit as the share falls, with the cap far away."""
    parents = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(8)]

    offspring = attempt_reproduction(parents, _share_config(share), np.random.default_rng(42))

    assert len(offspring) == expected


def test_recruitment_share_defaults_to_unlimited_recruitment() -> None:
    parents = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(8)]
    default = SimulationConfig(
        initial_population=8, max_population=100, recombination_probability=0.0, seed=42
    )

    assert default.reproduction_recruitment_share == pytest.approx(1.0)
    offspring = attempt_reproduction(parents, default, np.random.default_rng(42))
    assert len(offspring) == len(parents)


def test_a_scarce_recruitment_share_goes_to_the_most_accurate_parents() -> None:
    """With reproduction scarce, correctness rank decides who reproduces at all."""
    parents = _accurate_last()

    offspring = attempt_reproduction(parents, _share_config(0.5), np.random.default_rng(42))

    assert [child.state.parent_ids[0] for child in offspring] == [parents[3].id, parents[2].id]


def test_recruitment_share_never_starves_a_population_with_an_eligible_parent() -> None:
    """Rounding up keeps one recruit, so a small share cannot force extinction."""
    parents = [_adult(information=4.0, attention=4.0)]

    offspring = attempt_reproduction(parents, _share_config(0.01), np.random.default_rng(42))

    assert len(offspring) == 1


def test_recruitment_share_stays_within_the_population_cap() -> None:
    parents = [_adult(information=2.0 + i, attention=2.0 + i) for i in range(8)]
    capped = _share_config(1.0)
    capped = capped.model_copy(update={"max_population": 10})

    offspring = attempt_reproduction(parents, capped, np.random.default_rng(42))

    assert len(offspring) == 2


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
