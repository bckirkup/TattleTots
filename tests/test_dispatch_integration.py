"""Integration tests for COP dispatch loop and TattleTots layer."""

from __future__ import annotations

import pytest
from tattletots.engine.config import SimulationConfig
from tattletots.engine.dispatch_integration import init_user_cops, run_dispatch_cycle
from tattletots.engine.world import World
from tattletots.integration.tattletots_layer import TattleTotsLayer, resolve_layer
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


@pytest.mark.integration
class TestDispatchIntegration:
    def test_resolve_layer_names(self) -> None:
        assert resolve_layer("domain_only").name == "domain_only"
        assert resolve_layer("tattletots").name == "tattletots"

    def test_run_dispatch_cycle_with_scenario(self) -> None:
        config = SimulationConfig(initial_population=5, max_steps=10, seed=7)
        scenario = GaussianShiftScenario(total_steps=5, seed=7)
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)
        world.set_location_inference(scenario.infer_report_location)
        world.seed_population()
        cops = init_user_cops(world, scenario, config)

        for step in range(3):
            scenario.step(step)
            world.set_event_state(scenario.get_active_locations(step))
            world.step()
            outcomes, _ = run_dispatch_cycle(world, scenario, cops, step, config)
            assert isinstance(outcomes, list)

    def test_tattletots_layer_short_run(self) -> None:
        config = SimulationConfig(initial_population=5, max_steps=5, seed=3)
        scenario = GaussianShiftScenario(total_steps=5, seed=3)
        layer = TattleTotsLayer()
        run = _run_context(config, steps=2)
        state = layer.setup(scenario, run)
        for step in range(2):
            events = layer.step(scenario, step, state)
            assert "population" in events
        metrics = layer.finalize(scenario, state, run)
        assert "telemetry_summary" in metrics
        assert "simulation_output" in metrics


def _run_context(config: SimulationConfig, *, steps: int):
    from domain_runner.types import RunContext

    return RunContext(
        steps=steps,
        seed=config.seed or 42,
        domain_config={"steps": steps},
        layer="tattletots",
        simulation_config=config.model_dump(),
    )
