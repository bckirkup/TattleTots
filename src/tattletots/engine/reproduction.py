"""Reproduction and evolution: agents above energy threshold spawn offspring."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome
from tattletots.models.identity import seeded_id

_CORRECTNESS_PRIOR_REPORTS = 2.0
"""Pseudo-count of unrewarded reports shrinking a short reporting record toward zero."""


def attempt_reproduction(
    agents: list[Agent],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> list[Agent]:
    """Process reproduction for all eligible agents.

    Agents above the energy threshold reproduce (asexual or sexual).
    Population cap is enforced.
    """
    offspring: list[Agent] = []
    eligible = [a for a in agents if a.can_reproduce]

    if not eligible:
        return offspring

    if config.reproduction_merit_ordering:
        eligible = _merit_ordered(eligible, config)

    current_pop = len([a for a in agents if a.is_alive])

    for parent in eligible:
        if current_pop + len(offspring) >= config.max_population:
            break

        limiting_factor = parent.reproduction_limiting_factor(
            config.reproduction_coupling_strength,
            config.reproduction_information_scale,
            config.reproduction_attention_scale,
        )
        if limiting_factor < 1.0 and rng.random() >= limiting_factor:
            continue

        if rng.random() < config.recombination_probability and len(eligible) >= 2:
            # Sexual: pick a partner
            partners = [a for a in eligible if a.id != parent.id]
            if partners:
                partner = partners[rng.integers(0, len(partners))]
                child = _sexual_reproduction(parent, partner, config, rng)
                offspring.append(child)
                continue

        # Asexual reproduction
        child = parent.spawn_offspring(rng, mutation_rate=config.mutation_rate)
        offspring.append(child)

    return offspring


def _merit_ordered(eligible: list[Agent], config: SimulationConfig) -> list[Agent]:
    """Sort eligible parents by reproductive merit, best first.

    When the population cap binds it is the ordering, not the reserves, that decides
    who reproduces. Sorting by the same currency sufficiency that gates reproduction
    rations scarce opportunities by reserves instead of by agent creation order. With
    `reproduction_correctness_weight` above zero, part of that merit comes from rank in
    verified correctness instead, so the cap rations by what an agent got right rather
    than only by what it accumulated. The sort is stable, so creation order remains the
    tie-break.
    """

    def sufficiency(agent: Agent) -> float:
        return agent.reproduction_sufficiency(
            config.reproduction_information_scale,
            config.reproduction_attention_scale,
        )

    weight = config.reproduction_correctness_weight
    if weight <= 0.0:
        return sorted(eligible, key=sufficiency, reverse=True)

    reserve_rank = _fractional_ranks([sufficiency(agent) for agent in eligible])
    correctness_rank = _fractional_ranks([_verified_correctness(agent) for agent in eligible])
    merit = {
        agent.id: (1.0 - weight) * reserve_rank[index] + weight * correctness_rank[index]
        for index, agent in enumerate(eligible)
    }
    return sorted(eligible, key=lambda a: merit[a.id], reverse=True)


def _verified_correctness(agent: Agent) -> float:
    """Verified-correct share of an agent's reports, shrunk toward zero.

    The pseudo-count keeps an agent with one lucky correct report from outranking a
    sustained reporter, and leaves silent agents at zero rather than undefined.
    """
    reports = agent.state.reports_issued
    return agent.state.correct_reports / (reports + _CORRECTNESS_PRIOR_REPORTS)


def _fractional_ranks(values: list[float]) -> list[float]:
    """Rank each value in `[0, 1]`, largest value ranked 1.0, ties sharing a rank."""
    if len(values) < 2:
        return [1.0] * len(values)
    array = np.asarray(values, dtype=np.float64)
    positions = np.empty(array.size, dtype=np.float64)
    positions[np.argsort(array, kind="stable")] = np.arange(array.size, dtype=np.float64)
    unique_values, inverse = np.unique(array, return_inverse=True)
    inverse = inverse.reshape(-1)
    totals = np.zeros(unique_values.size, dtype=np.float64)
    np.add.at(totals, inverse, positions)
    counts = np.bincount(inverse, minlength=unique_values.size).astype(np.float64)
    tied = (totals / counts)[inverse] / (array.size - 1)
    return [float(rank) for rank in tied]


def _sexual_reproduction(
    parent_a: Agent,
    parent_b: Agent,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Agent:
    """Sexual recombination of two parents."""
    child_genome = Genome.recombine(parent_a.genome, parent_b.genome, rng)
    child_genome = child_genome.mutate(rng, rate=config.mutation_rate)

    # Both parents pay
    cost_per_parent = (
        parent_a.genome.reproduction_threshold + parent_b.genome.reproduction_threshold
    ) / 8
    parent_a.state.energy.information -= cost_per_parent
    parent_a.state.energy.attention -= cost_per_parent
    parent_b.state.energy.information -= cost_per_parent
    parent_b.state.energy.attention -= cost_per_parent

    return Agent(
        id=seeded_id(rng),
        genome=child_genome,
        state=AgentState(
            energy=EnergyReserves(
                information=cost_per_parent * 2,
                attention=cost_per_parent * 2,
            ),
            lifecycle=LifecycleStage.JUVENILE,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.state.generation, parent_b.state.generation) + 1,
        ),
    )
