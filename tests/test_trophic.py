"""Tests for label-keyed trophic attachment preferences."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine.config import SimulationConfig
from tattletots.engine.trophic import (
    compute_stream_attractiveness,
    select_input_streams,
    stream_attachment_key,
)
from tattletots.engine.world import World
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome
from tattletots.models.identity import stable_id_digest
from tattletots.models.stream import Stream, StreamType


def _stream(stream_id: str, label: str) -> Stream:
    return Stream(
        id=stream_id,
        stream_type=StreamType.RAW,
        dimensionality=2,
        label=label,
        current_data=np.array([0.0, 1.0]),
    )


def _agent(preference: np.ndarray) -> Agent:
    return Agent(genome=Genome(input_preference=preference))


def test_label_preference_raises_attachment_rate_with_margin() -> None:
    slot_count = 32
    x_stream = _stream("x-instance", "X")
    y_stream = _stream("y-instance", "Y")
    x_slot = stable_id_digest("X") % slot_count
    y_slot = stable_id_digest("Y") % slot_count
    assert x_slot != y_slot

    concentrated = np.zeros(slot_count)
    concentrated[x_slot] = 1.0
    flat = np.ones(slot_count)
    concentrated_count = 0
    flat_count = 0
    n_draws = 8 * 200
    for seed in range(8):
        concentrated_rng = np.random.default_rng(seed)
        flat_rng = np.random.default_rng(seed)
        concentrated_agent = _agent(concentrated)
        flat_agent = _agent(flat)
        for _ in range(200):
            if x_stream.id in select_input_streams(
                concentrated_agent,
                [x_stream, y_stream],
                max_inputs=1,
                rng=concentrated_rng,
            ):
                concentrated_count += 1
            if x_stream.id in select_input_streams(
                flat_agent,
                [x_stream, y_stream],
                max_inputs=1,
                rng=flat_rng,
            ):
                flat_count += 1

    concentrated_rate = concentrated_count / n_draws
    flat_rate = flat_count / n_draws
    margin = concentrated_rate - flat_rate
    assert concentrated_rate > 0.9
    assert flat_rate == pytest.approx(0.5, abs=0.05)
    assert margin > 0.25, f"label preference margin was only {margin:.3f}"


def test_same_label_streams_share_attachment_slot() -> None:
    first = _stream("first-instance", "X")
    second = _stream("second-instance", "X")
    preference = np.zeros(32)
    preference[stable_id_digest("X") % len(preference)] = 1.0
    agent = _agent(preference)
    rng = np.random.default_rng(7)

    assert stream_attachment_key(first) == stream_attachment_key(second) == "X"
    assert compute_stream_attractiveness(agent, first, rng) == pytest.approx(
        compute_stream_attractiveness(agent, second, rng)
    )


def test_empty_label_falls_back_to_stream_id() -> None:
    stream = _stream("empty-instance", "")
    preference = np.zeros(32)
    preference[stable_id_digest(stream.id) % len(preference)] = 1.0
    agent = _agent(preference)

    assert stream_attachment_key(stream) == stream.id
    assert compute_stream_attractiveness(agent, stream, np.random.default_rng(1)) > 0.0


def test_world_uses_configured_input_preference_slots() -> None:
    world = World(SimulationConfig(initial_population=2, input_preference_slots=37, seed=4))
    world.seed_population()

    assert {agent.genome.input_preference.size for agent in world.agents.values()} == {37}
