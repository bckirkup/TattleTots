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
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


def _stream(
    stream_id: str,
    label: str,
    stream_type: StreamType = StreamType.RAW,
) -> Stream:
    return Stream(
        id=stream_id,
        stream_type=stream_type,
        dimensionality=2,
        label=label,
        current_data=np.array([0.0, 1.0]),
        source_agent_id=None if stream_type == StreamType.RAW else f"src-{stream_id}",
    )


def _agent(preference: np.ndarray) -> Agent:
    return Agent(genome=Genome(input_preference=preference))


def _mixed_pool(n_raw: int = 3, n_residual: int = 27) -> list[Stream]:
    """A pool shaped like a running ecology: raw streams swamped by peer exhaust."""
    raw = [_stream(f"raw-{i}", f"raw_{i}", StreamType.RAW) for i in range(n_raw)]
    residual = [_stream(f"res-{i}", f"res_{i}", StreamType.RESIDUAL) for i in range(n_residual)]
    return raw + residual


def _grounded_attachment_rate(
    pool: list[Stream],
    *,
    grounded_fraction: float = 0.0,
    grounded_multiplier: float = 1.0,
    max_inputs: int = 3,
    n_seeds: int = 8,
    n_draws: int = 100,
) -> float:
    """Share of attached input slots that are grounded raw streams."""
    raw_ids = {s.id for s in pool if s.stream_type == StreamType.RAW}
    attached = 0
    slots = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        agent = _agent(np.ones(32))
        for _ in range(n_draws):
            chosen = select_input_streams(
                agent,
                pool,
                max_inputs=max_inputs,
                rng=rng,
                grounded_fraction=grounded_fraction,
                grounded_multiplier=grounded_multiplier,
            )
            attached += sum(1 for stream_id in chosen if stream_id in raw_ids)
            slots += len(chosen)
    return attached / slots


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


def test_grounded_fraction_grades_raw_attachment_share() -> None:
    pool = _mixed_pool()
    fractions = [0.0, 0.34, 0.67, 1.0]
    rates = [_grounded_attachment_rate(pool, grounded_fraction=fraction) for fraction in fractions]

    assert rates == sorted(rates), f"grounded_input_fraction is not monotone: {rates}"
    assert rates[0] < 0.2, f"unreserved raw share should reflect the pool mix: {rates[0]:.3f}"
    assert rates[-1] > 0.95, f"full reservation should saturate raw slots: {rates[-1]:.3f}"
    assert rates[1] - rates[0] > 0.15, f"one reserved slot moved too little: {rates}"
    assert all(0.0 <= rate <= 1.0 for rate in rates)


def test_grounded_multiplier_grades_raw_attachment_share() -> None:
    pool = _mixed_pool()
    multipliers = [1.0, 5.0, 25.0, 100.0]
    rates = [
        _grounded_attachment_rate(pool, grounded_multiplier=multiplier)
        for multiplier in multipliers
    ]

    assert rates == sorted(rates), f"grounded multiplier is not monotone: {rates}"
    assert rates[-1] - rates[0] > 0.2, f"multiplier looks dead: {rates}"
    assert all(0.0 <= rate <= 1.0 for rate in rates)


def test_default_knobs_preserve_unreserved_attachment_and_rng_stream() -> None:
    pool = _mixed_pool()
    agent_a = _agent(np.ones(32))
    agent_b = _agent(np.ones(32))
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)

    for _ in range(50):
        legacy = select_input_streams(agent_a, pool, max_inputs=3, rng=rng_a)
        defaulted = select_input_streams(
            agent_b,
            pool,
            max_inputs=3,
            rng=rng_b,
            grounded_fraction=0.0,
            grounded_multiplier=1.0,
        )
        assert legacy == defaulted
    assert rng_a.integers(0, 10**9) == rng_b.integers(0, 10**9)


def test_reservation_never_exceeds_available_raw_streams() -> None:
    pool = _mixed_pool(n_raw=1, n_residual=9)
    agent = _agent(np.ones(32))
    rng = np.random.default_rng(5)

    for _ in range(20):
        chosen = select_input_streams(agent, pool, max_inputs=3, rng=rng, grounded_fraction=1.0)
        assert len(chosen) == 3
        assert len(set(chosen)) == 3
        assert sum(1 for stream_id in chosen if stream_id.startswith("raw-")) == 1


def test_grounded_multiplier_does_not_change_residual_only_pool() -> None:
    residual_pool = _mixed_pool(n_raw=0, n_residual=10)
    baseline = _grounded_attachment_rate(residual_pool, n_seeds=2, n_draws=20)
    boosted = _grounded_attachment_rate(
        residual_pool, grounded_multiplier=50.0, n_seeds=2, n_draws=20
    )

    assert baseline == pytest.approx(0.0)
    assert boosted == pytest.approx(0.0)


def _world_raw_input_share(grounded_fraction: float, grounded_multiplier: float) -> float:
    adapter = GaussianShiftScenario(total_steps=60, seed=42)
    config = SimulationConfig(
        initial_population=20,
        max_population=40,
        max_steps=60,
        seed=11,
        grounded_input_fraction=grounded_fraction,
        grounded_attractiveness_multiplier=grounded_multiplier,
    )
    world = World(config=config)
    raw_ids = set()
    for stream in adapter.get_streams():
        world.add_stream(stream)
        raw_ids.add(stream.id)
    for user in adapter.get_users():
        world.add_user(user)
    world.seed_population()
    world.set_location_inference(adapter.infer_report_location)

    attached = 0
    slots = 0
    for step in range(60):
        adapter.step(step)
        world.set_event_state(adapter.get_active_locations(step))
        world.step()
        for agent in world.agents.values():
            if not agent.is_alive:
                continue
            ids = agent.state.input_stream_ids
            slots += len(ids)
            attached += sum(1 for stream_id in ids if stream_id in raw_ids)
    return attached / max(slots, 1)


def test_world_grounded_knobs_raise_raw_input_share() -> None:
    unreserved = _world_raw_input_share(0.0, 1.0)
    boosted = _world_raw_input_share(0.0, 50.0)
    reserved = _world_raw_input_share(0.5, 1.0)

    assert unreserved < reserved, f"reservation did not raise raw share: {unreserved} vs {reserved}"
    assert reserved > 0.4, f"reserved raw share too low: {reserved:.3f}"
    assert boosted > unreserved, f"multiplier did not raise raw share: {unreserved} vs {boosted}"
