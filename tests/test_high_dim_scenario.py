"""Tests for high-dimensional shift scenario."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.scenarios.high_dim_shift import HighDimShiftScenario


class TestHighDimScenario:
    def test_scenario_generates_1000_dim_stream(self) -> None:
        scenario = HighDimShiftScenario(dimensionality=1000, total_steps=50)
        scenario.step(0)
        streams = scenario.get_streams()
        assert streams[0].dimensionality == 1000
        assert streams[0].current_data.size == 1000

    def test_shift_localized_to_block(self) -> None:
        scenario = HighDimShiftScenario(
            dimensionality=1000,
            n_blocks=10,
            shift_block=3,
            shift_step=10,
            seed=0,
        )
        pre = scenario._generate_data(5)
        post = scenario._generate_data(15)
        block_size = 100
        start = 3 * block_size
        end = start + block_size
        pre_block_var = float(np.var(pre[start:end]))
        post_block_var = float(np.var(post[start:end]))
        assert post_block_var > pre_block_var

    def test_world_runs_with_high_dim_config(self) -> None:
        scenario = HighDimShiftScenario(dimensionality=200, n_blocks=5, total_steps=20)
        config = SimulationConfig(
            initial_population=5,
            max_population=20,
            max_stream_dim=64,
            max_working_dim=64,
            n_spatial_blocks=5,
        )
        world = World(config=config)
        for stream in scenario.get_streams():
            world.add_stream(stream)
        for user in scenario.get_users():
            world.add_user(user)
        world.set_dim_to_location(scenario.dim_index_to_location)
        world.seed_population()
        for step in range(5):
            scenario.step(step)
            world.set_event_state(scenario.get_active_locations(step))
            record = world.step()
            assert record.population >= 0

    def test_active_location_matches_shift_block(self) -> None:
        scenario = HighDimShiftScenario(shift_block=7, shift_step=100)
        locs = scenario.get_active_locations(100)
        assert locs == [(7, 0)]
