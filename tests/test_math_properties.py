"""Tests for §6 mathematical invariants not covered by test_smoke.py.

Covers:
  §6.2  Chain depth bounded by signal rank (~ceil(K/k))
  §6.3  Branching topologies more stable than linear
  §6.4  H-D-W equilibrium (honest-deceiver-whistleblower coexistence)
  §6.5  Domestication improves yield only when signals overlap
"""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.config import SimulationConfig
from tattletots.engine.domestication import apply_shaping, compute_shaping_signal
from tattletots.engine.trophic import compute_trophic_level
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
class TestChainDepthBound:
    """§6.2: Chain depth bounded by signal rank (~ceil(K/k))."""

    def test_trophic_depth_bounded_by_signal_rank(self) -> None:
        """Max trophic depth should not exceed ceil(K/k) + margin.

        With K=10 Gaussian components and typical k~3 PCA components,
        the theoretical max chain depth is ~ceil(10/3) ≈ 4. We allow
        generous margin for stochastic variation but depth should not
        be unbounded.
        """
        world = _run_scenario(steps=200, seed=42, population=25)
        levels = world.trophic_levels
        if not levels:
            pytest.skip("No living agents")

        max_depth = max(levels.values())

        # K=10 components, typical k=3 → theoretical bound ~4, with margin allow up to 8
        theoretical_bound = 8.0
        assert max_depth <= theoretical_bound, (
            f"Max trophic depth {max_depth:.1f} exceeds theoretical bound "
            f"{theoretical_bound:.1f} for K=10 Gaussian scenario"
        )

    def test_chain_depth_from_topology(self) -> None:
        """Verify compute_trophic_level follows input graph topology correctly."""
        # Construct a known topology: A -> B -> C (chain of depth 3)
        agent_inputs: dict[str, list[str]] = {
            "agent_a": ["raw_stream_1"],
            "agent_b": ["residual_a"],
            "agent_c": ["residual_b"],
        }
        stream_sources: dict[str, str | None] = {
            "raw_stream_1": None,
            "residual_a": "agent_a",
            "residual_b": "agent_b",
        }

        level_a = compute_trophic_level("agent_a", agent_inputs, stream_sources)
        level_b = compute_trophic_level("agent_b", agent_inputs, stream_sources)
        level_c = compute_trophic_level("agent_c", agent_inputs, stream_sources)

        assert level_a == pytest.approx(1.0)
        assert level_b == pytest.approx(2.0)
        assert level_c == pytest.approx(3.0)


@pytest.mark.smoke
class TestBranchingStability:
    """§6.3: Branching topologies more stable than linear."""

    def test_branching_more_stable_than_linear(self) -> None:
        """Verify that branching topologies tolerate node removal better.

        We construct two topologies from the trophic level computation:
        - Branching: agent C consumes both A's and B's residuals
        - Linear: agent C depends solely on B, which depends on A

        When the sole bridge node is removed from the linear chain,
        the downstream agent loses all inputs. In the branching case,
        removing one upstream still leaves the other.
        """
        # Branching: C depends on {A, B}; remove A → C still has B
        branch_inputs: dict[str, list[str]] = {
            "A": ["raw1"],
            "B": ["raw2"],
            "C": ["resA", "resB"],
        }
        branch_sources: dict[str, str | None] = {
            "raw1": None,
            "raw2": None,
            "resA": "A",
            "resB": "B",
        }
        # Remove A's residual
        branch_inputs_after = {k: v for k, v in branch_inputs.items() if k != "A"}
        del branch_sources["resA"]

        level_c_after = compute_trophic_level("C", branch_inputs_after, branch_sources)
        # C still has input via B → still computable
        assert level_c_after > 1.0, "C should still have a trophic level after losing A"

        # Linear: C depends on B depends on A; remove B → C has no input
        linear_inputs: dict[str, list[str]] = {
            "A": ["raw1"],
            "B": ["resA"],
            "C": ["resB"],
        }
        linear_sources: dict[str, str | None] = {
            "raw1": None,
            "resA": "A",
            "resB": "B",
        }
        # Remove B's residual (bridge node dies)
        linear_inputs_after = {k: v for k, v in linear_inputs.items() if k != "B"}
        del linear_sources["resB"]

        level_c_linear = compute_trophic_level("C", linear_inputs_after, linear_sources)
        # C has no inputs → falls back to base level 1
        assert level_c_linear == pytest.approx(
            1.0
        ), "C should lose its trophic position when bridge node B is removed"

        # Branching C retains higher level than linear C after disruption
        assert level_c_after > level_c_linear


@pytest.mark.smoke
class TestHDWEquilibrium:
    """§6.4: Honest-Deceiver-Whistleblower equilibrium.

    In a mixed population, honest reporters (accurate escalation),
    deceivers (low threshold, many false alarms), and whistleblowers
    (agents that detect dishonesty) should coexist rather than one
    strategy dominating entirely.
    """

    def test_multiple_strategies_coexist(self) -> None:
        """Verify that no single escalation strategy dominates completely.

        We seed agents with diverse escalation thresholds (proxy for
        honest/deceiver spectrum) and check that both conservative
        and aggressive strategies survive.
        """
        config = SimulationConfig(
            initial_population=2,
            max_population=60,
            max_steps=200,
            seed=42,
            mutation_rate=0.15,
            false_alarm_penalty=0.3,
        )
        scenario = GaussianShiftScenario(shift_step=100, total_steps=200, seed=42)
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)

        world.seed_population()

        rng = np.random.default_rng(42)
        # Add 20 agents with diverse escalation thresholds
        for _ in range(10):
            # "Honest" — high threshold, conservative
            genome_h = Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=float(rng.uniform(0.6, 0.9)),
                compute_cost=0.08,
                maintenance_cost=0.03,
                reproduction_threshold=2.0,
            )
            agent = Agent(
                genome=genome_h,
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    energy=EnergyReserves(information=3.0, attention=3.0),
                ),
            )
            world.agents[agent.id] = agent
            world._init_agent_model(agent)

        for _ in range(10):
            # "Deceiver" — low threshold, aggressive
            genome_d = Genome(
                compression_type=CompressionType.THRESHOLD,
                escalation_threshold=float(rng.uniform(0.05, 0.2)),
                compute_cost=0.08,
                maintenance_cost=0.03,
                reproduction_threshold=2.0,
            )
            agent = Agent(
                genome=genome_d,
                state=AgentState(
                    lifecycle=LifecycleStage.ADULT,
                    energy=EnergyReserves(information=3.0, attention=3.0),
                ),
            )
            world.agents[agent.id] = agent
            world._init_agent_model(agent)

        for step_num in range(200):
            scenario.step(step_num)
            world.set_ground_truth(scenario.get_ground_truth(step_num))
            world.step()
            if world.living_population == 0:
                break

        living = [a for a in world.agents.values() if a.is_alive]
        if len(living) < 3:
            pytest.skip("Population too small for equilibrium check")

        thresholds = [a.genome.escalation_threshold for a in living]
        threshold_range = max(thresholds) - min(thresholds)

        # Diversity should persist — range should be > 0.1
        assert threshold_range > 0.1, (
            f"Escalation threshold range {threshold_range:.3f} too narrow; "
            "expected diverse strategies to coexist"
        )


@pytest.mark.smoke
class TestDomesticationOverlap:
    """§6.5: Domestication improves yield only when signals overlap."""

    def test_domestication_with_overlapping_signals(self) -> None:
        """Shaping should modify preferences when signal dimensions match."""
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )

        # Downstream signal overlaps in dimensionality
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )

        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])

        # Override should have been written to state
        assert upstream.state.input_preference_override.size > 0
        # Preferences should have shifted toward the shaping signal
        override = upstream.state.input_preference_override
        assert (
            override[0] > 0.5
        ), f"First preference {override[0]:.3f} should have increased toward downstream signal"

    def test_domestication_without_overlap(self) -> None:
        """Shaping should have no effect when dimensions don't match."""
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )

        # Downstream signal has DIFFERENT dimensionality (2 vs 3)
        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.2]),
            ),
        )

        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])

        # Override should NOT have been written (dim mismatch → early return)
        assert upstream.state.input_preference_override.size == 0

    def test_domestication_zero_sensitivity(self) -> None:
        """Agents with zero domestication_sensitivity should be immune."""
        genome = Genome(
            input_preference=np.array([0.5, 0.3, 0.2]),
            domestication_sensitivity=0.0,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )

        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )

        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])

        # No override — sensitivity is zero
        assert upstream.state.input_preference_override.size == 0

    def test_domestication_does_not_mutate_genome(self) -> None:
        """Domestication must write to state, not genome (genome immutability)."""
        original_pref = np.array([0.5, 0.3, 0.2])
        genome = Genome(
            input_preference=original_pref.copy(),
            domestication_sensitivity=0.5,
        )
        upstream = Agent(
            genome=genome,
            state=AgentState(lifecycle=LifecycleStage.ADULT),
        )

        downstream = Agent(
            state=AgentState(
                lifecycle=LifecycleStage.ADULT,
                signal_vector=np.array([0.8, 0.1, 0.1]),
            ),
        )

        signal = compute_shaping_signal(downstream, upstream)
        apply_shaping(upstream, [signal])

        # Genome should be unchanged
        np.testing.assert_array_almost_equal(
            upstream.genome.input_preference,
            original_pref,
            err_msg="Domestication must not mutate genome.input_preference",
        )
