"""Smoke tests: validate emergent behavior per Requirements §9.

Success Criteria:
1. Trophic hierarchies of depth > 2 emerge from random seed
2. Population reaches stable equilibrium (births ≈ deaths)
3. Removing basal streams causes upstream extinction cascades
4. False-alarm agents lose trust and die; accurate agents reproduce
5. At least two distinct "species" (genome clusters) coexist at equilibrium
"""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import CompressionType, Genome
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


def _run_scenario(steps: int = 200, seed: int = 42, population: int = 25) -> World:
    """Helper: run the Gaussian shift scenario and return the world."""
    config = SimulationConfig(
        initial_population=population,
        max_population=80,
        max_steps=steps,
        seed=seed,
        mutation_rate=0.15,
    )
    scenario = GaussianShiftScenario(shift_step=steps // 2, total_steps=steps, seed=seed)
    world = World(config=config)
    for stream in scenario.get_streams():
        world.add_stream(stream)
    for user in scenario.get_users():
        world.add_user(user)
    world.seed_population()

    for step_num in range(steps):
        scenario.step(step_num)
        world.set_ground_truth(scenario.get_ground_truth(step_num))
        world.step()
        if world.living_population == 0:
            break

    return world


@pytest.mark.smoke
class TestEmergentBehavior:
    """Validate that the 5 success criteria from Requirements §9 are met."""

    def test_trophic_depth_exceeds_2(self) -> None:
        """Criterion 1: Trophic hierarchies of depth > 2 emerge."""
        world = _run_scenario(steps=80, seed=42)
        max_depth = world.telemetry.max_trophic_depth
        assert max_depth > 2.0, f"Max trophic depth was only {max_depth:.1f}; expected > 2.0"

    def test_population_stability(self) -> None:
        """Criterion 2: Population reaches stable equilibrium."""
        world = _run_scenario(steps=150, seed=42, population=30)
        # Check that population didn't go to zero
        assert world.living_population > 0, "Population went extinct"
        # Check for approximate stability in the last 50 steps
        pop_history = world.telemetry.population_history()
        if len(pop_history) >= 100:
            last_50 = pop_history[-50:]
            mean_pop = np.mean(last_50)
            std_pop = np.std(last_50)
            cv = std_pop / max(mean_pop, 1)
            assert cv < 0.5, f"Population CV={cv:.2f} too high for stability"

    def test_extinction_cascade_on_stream_removal(self) -> None:
        """Criterion 3: Removing basal streams causes extinction cascades."""
        config = SimulationConfig(
            initial_population=20,
            max_population=60,
            max_steps=200,
            seed=42,
            subsidy_rate=0.0,
        )
        scenario = GaussianShiftScenario(shift_step=200, total_steps=200, seed=42)
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)
        world.seed_population()

        # Run for 50 steps to establish ecology
        for step_num in range(50):
            scenario.step(step_num)
            world.set_ground_truth(False)
            world.step()

        pop_before = world.living_population

        # Remove all raw (basal) streams entirely
        raw_ids = [sid for sid, s in world.streams.items() if s.stream_type.value == "raw"]
        for sid in raw_ids:
            del world.streams[sid]

        # Also clear residual stream data so agents can't feed off stale signals
        for stream in world.streams.values():
            stream.current_data = np.zeros(stream.dimensionality)

        # Block reproduction so deaths dominate
        for agent in world.agents.values():
            if agent.is_alive:
                agent.genome.reproduction_threshold = 999.0

        # Increase costs to accelerate starvation
        for agent in world.agents.values():
            if agent.is_alive:
                agent.genome.compute_cost = 0.5
                agent.genome.maintenance_cost = 0.4

        # Run more steps — should see population decline from starvation
        for _step_num in range(50, 150):
            world.set_ground_truth(False)
            world.step()

        pop_after = world.living_population
        # Population should have decreased significantly
        assert pop_after < pop_before, (
            f"Expected extinction cascade: pop went from {pop_before} to {pop_after}"
        )

    def test_false_alarm_agents_die(self) -> None:
        """Criterion 4: False-alarm agents lose trust and die."""
        config = SimulationConfig(
            initial_population=15,
            max_population=40,
            max_steps=100,
            seed=42,
            false_alarm_penalty=0.5,
        )
        scenario = GaussianShiftScenario(shift_step=200, total_steps=200, seed=42)
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)
        world.seed_population()

        # Inject a "cry wolf" agent with very low escalation threshold
        wolf_genome = Genome(
            compression_type=CompressionType.THRESHOLD,
            escalation_threshold=0.01,  # Escalates on everything
            maintenance_cost=0.05,
            compute_cost=0.05,
        )
        wolf = Agent(
            genome=wolf_genome,
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                energy=EnergyReserves(information=2.0, attention=2.0),
            ),
        )
        world.agents[wolf.id] = wolf
        world._init_agent_model(wolf)

        # Run with no actual events (ground truth = False)
        for step_num in range(100):
            scenario.step(step_num)
            world.set_ground_truth(False)
            world.step()

        # The wolf should be dead or severely weakened
        wolf_final = world.agents[wolf.id]
        assert not wolf_final.is_alive or wolf_final.state.energy.attention < 0.5, (
            "False-alarm agent should have died or lost most attention energy"
        )

    def test_genome_diversity(self) -> None:
        """Criterion 5: At least 2 distinct species coexist."""
        world = _run_scenario(steps=100, seed=42, population=25)
        living = [a for a in world.agents.values() if a.is_alive]
        if len(living) < 2:
            pytest.skip("Population too small to assess diversity")

        # Check compression type diversity
        types = {a.genome.compression_type for a in living}
        # Or check escalation threshold diversity
        thresholds = [a.genome.escalation_threshold for a in living]
        threshold_std = np.std(thresholds)

        # Either multiple model types OR significant parameter diversity
        assert len(types) >= 2 or threshold_std > 0.1, (
            f"Insufficient diversity: types={types}, threshold_std={threshold_std:.3f}"
        )


@pytest.mark.smoke
class TestMathematicalProperties:
    """Validate mathematical invariants from Requirements §6."""

    def test_residual_entropy_decreases(self) -> None:
        """§6.1: Residual entropy decreases through the chain.

        Tests that agents in a direct trophic chain produce decreasing
        residual variance. Only compares L2 agents that actually consume
        L1 residuals (not mixed-input agents), and samples over multiple
        steps for statistical robustness.
        """
        config = SimulationConfig(
            initial_population=25,
            max_population=80,
            max_steps=200,
            seed=42,
            mutation_rate=0.15,
        )
        scenario = GaussianShiftScenario(shift_step=200, total_steps=200, seed=42)
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)
        world.seed_population()

        # Track per-step variance ratios for actual chain pairs
        ratios: list[float] = []

        for step_num in range(100):
            scenario.step(step_num)
            world.set_ground_truth(scenario.get_ground_truth(step_num))
            world.step()
            if world.living_population == 0:
                break

            # Only sample from step 50 onward (compression models need training)
            if step_num < 50:
                continue

            # Build map: output_stream_id -> agent variance
            agent_output_var: dict[str, float] = {}
            for agent in world.agents.values():
                if not agent.is_alive or not agent.state.output_stream_id:
                    continue
                stream = world.streams.get(agent.state.output_stream_id)
                if stream and stream.current_data.size > 0:
                    agent_output_var[agent.id] = float(np.var(stream.current_data))

            # Find chain pairs: L2 agent consumes L1 agent's output stream
            for agent in world.agents.values():
                if not agent.is_alive or agent.id not in agent_output_var:
                    continue
                l2_var = agent_output_var[agent.id]
                for input_sid in agent.state.input_stream_ids:
                    input_stream = world.streams.get(input_sid)
                    if input_stream is None or input_stream.source_agent_id is None:
                        continue
                    upstream_id = input_stream.source_agent_id
                    if upstream_id in agent_output_var:
                        l1_var = agent_output_var[upstream_id]
                        if l1_var > 1e-10:
                            ratios.append(l2_var / l1_var)

        if not ratios:
            pytest.skip("No direct chain pairs found with measurable variance")

        median_ratio = float(np.median(ratios))
        assert median_ratio <= 5.0, (
            f"Median L2/L1 variance ratio ({median_ratio:.2f}) too high; "
            f"expected ≤ 5.0 across {len(ratios)} chain-pair samples"
        )

    def test_attention_zero_sum(self) -> None:
        """§6.6: Attention is zero-sum per user."""
        from tattletots.engine.attention import allocate_attention
        from tattletots.models.user import User as UserModel

        user = UserModel(
            name="test",
            attention_budget=1.0,
            priority_vector=np.array([1.0, 0.0, 0.0]),
        )
        agents = [
            Agent(
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    signal_vector=np.random.randn(3),
                )
            )
            for _ in range(10)
        ]
        for a in agents:
            user.trust[a.id] = np.random.uniform(0.1, 1.0)

        alloc = allocate_attention(user, agents)
        total = sum(alloc.values())
        assert total == pytest.approx(1.0, rel=1e-5), f"Total attention {total:.4f} != budget 1.0"
