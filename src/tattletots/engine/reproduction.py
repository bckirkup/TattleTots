"""Reproduction and evolution: agents above energy threshold spawn offspring."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome


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

    current_pop = len([a for a in agents if a.is_alive])

    for parent in eligible:
        if current_pop + len(offspring) >= config.max_population:
            break

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


def _sexual_reproduction(
    parent_a: Agent,
    parent_b: Agent,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Agent:
    """Sexual recombination of two parents."""
    child_genome = Genome.recombine(parent_a.genome, parent_b.genome, rng)
    child_genome = child_genome.mutate(rng, rate=config.mutation_rate)

    # Both parents pay (based on parents' thresholds, not child's mutated one)
    cost_per_parent = (
        parent_a.genome.reproduction_threshold + parent_b.genome.reproduction_threshold
    ) / 8
    parent_a.state.energy.information -= cost_per_parent
    parent_a.state.energy.attention -= cost_per_parent
    parent_b.state.energy.information -= cost_per_parent
    parent_b.state.energy.attention -= cost_per_parent

    return Agent(
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
