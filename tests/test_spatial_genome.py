"""Tests for heritable generic spatial inference."""

from __future__ import annotations

import numpy as np

from tattletots.engine.config import SimulationConfig
from tattletots.engine.spatial import infer_geometry_location
from tattletots.engine.world import World
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, SpatialInferenceStrategy
from tattletots.models.observation import ObservationPacket, StreamMetadata
from tattletots.models.stream import Stream
from tattletots.models.user import User
from tattletots.scenarios.sparse_sensor import SparseSensorScenario


def _geometry_packet() -> ObservationPacket:
    return ObservationPacket(
        data=np.array([0.2, 3.0, 0.5]),
        metadata=StreamMetadata(
            coordinates=[(0.0, 0.0), (4.0, 2.0), (8.0, 8.0)],
            modality=["a", "a", "a"],
        ),
        status=np.array(["observed", "observed", "observed"], dtype="<U8"),
    )


def test_spatial_traits_mutate_and_span_strategies() -> None:
    parent = Genome(spatial_region=(0, 0), spatial_radius=0)
    children = [parent.mutate(np.random.default_rng(seed), rate=1.0) for seed in range(8)]

    assert all(child.spatial_inference_strategy in SpatialInferenceStrategy for child in children)
    assert len({child.spatial_inference_strategy for child in children}) > 1
    assert any(
        child.spatial_inference_strategy != parent.spatial_inference_strategy for child in children
    )


def test_random_founders_span_competent_and_incompetent_strategies() -> None:
    genomes = [
        Genome.random_genome(np.random.default_rng(seed), n_streams=1, n_users=1)
        for seed in range(100)
    ]

    strategies = {genome.spatial_inference_strategy for genome in genomes}
    assert len(strategies) >= 3
    assert SpatialInferenceStrategy.FIXED_PRIOR in strategies
    assert SpatialInferenceStrategy.KERNEL in strategies


def test_spatial_traits_recombine_from_parent_genomes() -> None:
    parent_a = Genome(
        spatial_region=(0, 0),
        spatial_inference_strategy=SpatialInferenceStrategy.PEAK,
        spatial_kernel_bandwidth=1.0,
    )
    parent_b = Genome(
        spatial_region=(9, 9),
        spatial_inference_strategy=SpatialInferenceStrategy.KERNEL,
        spatial_kernel_bandwidth=9.0,
    )

    children = [
        Genome.recombine(parent_a, parent_b, np.random.default_rng(seed)) for seed in range(8)
    ]

    assert {child.spatial_inference_strategy for child in children} <= {
        SpatialInferenceStrategy.PEAK,
        SpatialInferenceStrategy.KERNEL,
    }
    assert {child.spatial_kernel_bandwidth for child in children} <= {1.0, 9.0}


def test_geometry_inference_uses_genome_strategy() -> None:
    packet = _geometry_packet()
    peak_agent = Agent(genome=Genome(spatial_inference_strategy=SpatialInferenceStrategy.PEAK))
    fixed_agent = Agent(
        genome=Genome(
            spatial_inference_strategy=SpatialInferenceStrategy.FIXED_PRIOR,
            spatial_region=(7, 6),
        )
    )

    assert infer_geometry_location(peak_agent, packet) == (4, 2)
    assert infer_geometry_location(fixed_agent, packet) == (7, 6)


def test_geometry_location_precedes_callback_but_metadata_free_keeps_callback() -> None:
    world = World(config=SimulationConfig(initial_population=2, seed=42))
    user = User(name="user", attention_budget=1.0, priority_vector=np.ones(1))
    world.add_user(user)
    agent = Agent()
    stream = Stream(dimensionality=3, current_data=np.ones(3), label="legacy")
    world.add_stream(stream)
    agent.state.input_stream_ids = [stream.id]
    agent.state.last_geometry_location = (4, 2)
    agent.state.last_inferred_location = (1, 1)
    world.set_location_inference(lambda _data, _labels: (9, 9))

    assert world._resolve_report_location(agent, np.ones(3)) == (4, 2)
    agent.state.last_geometry_location = None
    assert world._resolve_report_location(agent, np.ones(3)) == (9, 9)


def test_sparse_sensor_ground_truth_is_reachable_from_published_streams() -> None:
    scores: list[tuple[float, float, float]] = []
    for seed in (7, 42, 99, 123, 2024):
        scenario = SparseSensorScenario(seed=seed)
        max_correct = inverse_correct = bound_correct = 0
        for step in range(scenario.total_steps):
            scenario.step(step)
            truth = tuple(scenario.get_active_locations(step)[0])
            max_correct += scenario.reference_max_signal() == truth
            inverse_correct += scenario.reference_inverse_distance() == truth
            bound_correct += scenario.achievable_location_bound() == truth
        denominator = float(scenario.total_steps)
        scores.append(
            (max_correct / denominator, inverse_correct / denominator, bound_correct / denominator)
        )

    mean_scores = np.mean(scores, axis=0)
    assert mean_scores[0] > 0.05
    assert mean_scores[2] > mean_scores[0] + 0.1
    assert mean_scores[2] > 0.2
