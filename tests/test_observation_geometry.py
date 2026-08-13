"""Tests for domain-declared observation metadata transport."""

from __future__ import annotations

import numpy as np
import pytest

from tattletots.engine import sensing, temporal
from tattletots.engine.config import SimulationConfig
from tattletots.engine.sensing import prepare_agent_observation
from tattletots.engine.spatial import apply_spatial_observation
from tattletots.engine.temporal import apply_temporal_observation
from tattletots.engine.world import World
from tattletots.models.agent import Agent, AgentState
from tattletots.models.genome import Genome, SensingStrategy, SpatialStrategy, TemporalFusionMode
from tattletots.models.observation import ObservationPacket, ObservationStatus, StreamMetadata
from tattletots.models.stream import Stream, StreamType


def _metadata(size: int, prefix: str = "sensor") -> StreamMetadata:
    return StreamMetadata(
        coordinates=[(float(i), float(i + 1)) for i in range(size)],
        modality=[prefix] * size,
        identity=[f"{prefix}-{i}" for i in range(size)],
        footprints=[(1.0, 1.0)] * size,
        resolution=[1.0] * size,
    )


def _agent(streams: list[Stream], **genome_kwargs: object) -> Agent:
    return Agent(
        genome=Genome(working_dim=8, **genome_kwargs),
        state=AgentState(input_stream_ids=[stream.id for stream in streams]),
    )


def test_missing_status_is_distinct_from_zero_value() -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.zeros(2),
        metadata=_metadata(2),
    )
    stream.update(
        np.zeros(2),
        [ObservationStatus.MISSING, ObservationStatus.OBSERVED],
    )

    assert stream.current_data[0] == stream.current_data[1] == 0.0
    assert list(stream.current_status) == ["missing", "observed"]


def test_sensor_geometry_survives_missing_status_and_selection() -> None:
    metadata = StreamMetadata(
        sensor_coordinates=[(0.0, 0.0), (4.0, 2.0)],
        modality=["point", "point"],
    )
    selected = metadata.select(np.array([1], dtype=np.int64))

    assert selected.coordinates is None
    assert selected.sensor_coordinates == [(4.0, 2.0)]


def test_concat_truncation_preserves_feature_metadata() -> None:
    first = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.array([1.0, 2.0]),
        metadata=_metadata(2, "first"),
    )
    second = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.array([3.0, 4.0]),
        metadata=_metadata(2, "second"),
    )
    agent = _agent([first, second], sensing_strategy=SensingStrategy.CONCAT)

    observation, _, _, _ = prepare_agent_observation(
        agent,
        {first.id: first, second.id: second},
        SimulationConfig(max_stream_dim=3),
    )

    assert observation.metadata is not None
    assert observation.metadata.coordinates == [(0.0, 1.0), (1.0, 2.0), (0.0, 1.0)]
    assert list(observation.metadata.modality or []) == ["first", "first", "second"]
    assert observation.status is not None
    assert observation.status.size == 3


def test_subspace_selection_preserves_selected_metadata() -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=6,
        current_data=np.arange(6.0),
        metadata=_metadata(6),
    )
    agent = _agent(
        [stream],
        sensing_strategy=SensingStrategy.SUBSPACE_SAMPLE,
        dim_offset=3,
    )
    observation, _, _, _ = prepare_agent_observation(
        agent,
        {stream.id: stream},
        SimulationConfig(max_stream_dim=3),
    )

    assert observation.metadata is not None
    assert len(observation.metadata.coordinates or []) == observation.data.size
    assert set(observation.metadata.identity or []) <= {
        "sensor-0",
        "sensor-1",
        "sensor-2",
        "sensor-3",
        "sensor-4",
        "sensor-5",
    }


@pytest.mark.parametrize(
    ("strategy", "working_dim", "n_blocks"),
    [
        (SensingStrategy.CONCAT, 8, 10),
        (SensingStrategy.SUBSPACE_SAMPLE, 8, 10),
        (SensingStrategy.BLOCK_SPECIALIZE, 8, 3),
        (SensingStrategy.WEIGHTED_FUSE, 8, 10),
    ],
)
def test_metadata_and_status_lengths_match_numeric_selection(
    strategy: SensingStrategy,
    working_dim: int,
    n_blocks: int,
) -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=10,
        current_data=np.arange(10.0),
        metadata=_metadata(10),
    )
    agent = _agent([stream], sensing_strategy=strategy)
    observation, _, _, _ = prepare_agent_observation(
        agent,
        {stream.id: stream},
        SimulationConfig(max_stream_dim=working_dim, n_spatial_blocks=n_blocks),
    )

    if observation.metadata is None:
        assert observation.status is None
        return
    assert observation.metadata.feature_count == observation.data.size
    assert observation.status is not None
    assert observation.status.size == observation.data.size


def test_metadata_consumes_numeric_selection_without_rederiving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=6,
        current_data=np.arange(6.0),
        metadata=_metadata(6),
    )
    agent = _agent(
        [stream],
        sensing_strategy=SensingStrategy.SUBSPACE_SAMPLE,
    )
    selections = [np.array([1, 4, 5], dtype=np.int64)]

    def select_once(total_dim: int, working_dim: int, seed: int) -> np.ndarray:
        assert total_dim == 6
        assert working_dim == 8
        return selections.pop()

    monkeypatch.setattr(sensing, "_stable_sample_indices", select_once)
    observation, _, _, _ = sensing.prepare_agent_observation(
        agent,
        {stream.id: stream},
        SimulationConfig(max_stream_dim=8),
    )

    assert selections == []
    assert observation.data[:3].tolist() == [1.0, 4.0, 5.0]
    assert observation.metadata is not None
    assert observation.metadata.identity[:3] == ["sensor-1", "sensor-4", "sensor-5"]


def test_weighted_fusion_drops_geometry_after_provenance_is_combined() -> None:
    first = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.ones(2),
        metadata=_metadata(2, "first"),
    )
    second = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.ones(2),
        metadata=_metadata(2, "second"),
    )
    agent = _agent(
        [first, second],
        sensing_strategy=SensingStrategy.WEIGHTED_FUSE,
        fusion_weights=np.array([0.5, 0.5]),
    )

    observation, _, _, _ = prepare_agent_observation(
        agent,
        {first.id: first, second.id: second},
        SimulationConfig(max_stream_dim=2),
    )

    assert observation.metadata is None


def test_spatial_mask_marks_features_as_masked_but_keeps_geometry() -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=5,
        current_data=np.array([0.0, 0.0, 2.0, 0.0, 0.0]),
        metadata=_metadata(5),
    )
    agent = _agent(
        [stream],
        spatial_strategy=SpatialStrategy.PEAK,
    )
    observation, _, _, _ = prepare_agent_observation(
        agent,
        {stream.id: stream},
        SimulationConfig(max_stream_dim=5),
    )
    masked = apply_spatial_observation(agent, observation)

    assert masked.metadata == observation.metadata
    assert masked.status is not None
    assert list(masked.status) == ["masked", "observed", "observed", "observed", "masked"]


@pytest.mark.parametrize(
    "mode",
    [
        TemporalFusionMode.EMA,
        TemporalFusionMode.WINDOW_STACK,
        TemporalFusionMode.AR_LAG,
    ],
)
def test_temporal_fusion_retains_geometry_when_schema_agrees(
    mode: TemporalFusionMode,
) -> None:
    stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.ones(2),
        metadata=_metadata(2),
    )
    agent = _agent(
        [stream],
        temporal_memory_depth=2,
        temporal_fusion_mode=mode,
    )
    observation, _, _, _ = prepare_agent_observation(
        agent,
        {stream.id: stream},
        SimulationConfig(max_stream_dim=2),
    )

    first = apply_temporal_observation(agent, observation)
    second = apply_temporal_observation(agent, observation)

    assert first.metadata is not None
    assert second.metadata == first.metadata
    assert second.status is not None
    assert list(second.status) == ["observed", "observed"]
    assert second.observed_fraction is not None
    np.testing.assert_allclose(second.observed_fraction, [1.0, 1.0])


def test_temporal_fusion_drops_geometry_when_schema_changes() -> None:
    first_stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.ones(2),
        metadata=_metadata(2, "first"),
    )
    second_stream = Stream(
        stream_type=StreamType.RAW,
        dimensionality=2,
        current_data=np.ones(2),
        metadata=_metadata(2, "second"),
    )
    agent = _agent(
        [first_stream],
        temporal_memory_depth=2,
        temporal_fusion_mode=TemporalFusionMode.EMA,
    )
    first, _, _, _ = prepare_agent_observation(
        agent,
        {first_stream.id: first_stream},
        SimulationConfig(max_stream_dim=2),
    )
    second = ObservationPacket(
        data=second_stream.current_data,
        metadata=second_stream.metadata,
        status=second_stream.current_status,
    )

    assert apply_temporal_observation(agent, first).metadata is not None
    fused = apply_temporal_observation(agent, second)

    assert fused.metadata is None
    assert fused.observed_fraction is None
    assert len(agent.state.temporal_buffer) == 2


def test_temporal_fusion_resets_history_when_dimensionality_changes() -> None:
    agent = Agent(
        genome=Genome(
            working_dim=8,
            temporal_memory_depth=2,
            temporal_fusion_mode=TemporalFusionMode.EMA,
        )
    )
    first = ObservationPacket(data=np.ones(2), metadata=_metadata(2))
    second = ObservationPacket(data=np.ones(3), metadata=_metadata(3))

    apply_temporal_observation(agent, first)
    fused = apply_temporal_observation(agent, second)

    assert fused.data.size == 3
    assert fused.metadata is None
    assert fused.observed_fraction is None
    assert len(agent.state.temporal_buffer) == 1


def test_temporal_status_preserves_alternating_missing_signal() -> None:
    agent = Agent(
        genome=Genome(
            working_dim=8,
            temporal_memory_depth=2,
            temporal_fusion_mode=TemporalFusionMode.EMA,
        )
    )
    metadata = _metadata(2)
    observed = ObservationPacket(
        data=np.ones(2),
        metadata=metadata,
        status=np.array(["observed", "observed"], dtype="<U8"),
    )
    missing = ObservationPacket(
        data=np.ones(2),
        metadata=metadata,
        status=np.array(["missing", "observed"], dtype="<U8"),
    )

    apply_temporal_observation(agent, observed)
    fused = apply_temporal_observation(agent, missing)

    assert fused.metadata == metadata
    assert fused.status is not None
    assert list(fused.status) == ["missing", "observed"]
    assert fused.observed_fraction is not None
    np.testing.assert_allclose(fused.observed_fraction, [0.5, 1.0])


@pytest.mark.parametrize(
    ("statuses", "expected_status", "expected_fraction"),
    [
        (
            [["missing", "missing"], ["missing", "missing"]],
            ["missing", "missing"],
            [0.0, 0.0],
        ),
        (
            [["observed", "missing"], ["missing", "observed"]],
            ["missing", "missing"],
            [0.5, 0.5],
        ),
        (
            [["observed", "observed"], ["observed", "observed"]],
            ["observed", "observed"],
            [1.0, 1.0],
        ),
    ],
)
def test_temporal_observed_fraction_distinguishes_availability_histories(
    statuses: list[list[str]],
    expected_status: list[str],
    expected_fraction: list[float],
) -> None:
    agent = Agent(
        genome=Genome(
            working_dim=8,
            temporal_memory_depth=2,
            temporal_fusion_mode=TemporalFusionMode.EMA,
        )
    )
    metadata = _metadata(2)
    for values in statuses:
        packet = ObservationPacket(
            data=np.ones(2),
            metadata=metadata,
            status=np.asarray(values, dtype="<U8"),
        )
        fused = apply_temporal_observation(agent, packet)

    assert fused.metadata == metadata
    assert fused.status is not None
    assert list(fused.status) == expected_status
    assert fused.observed_fraction is not None
    np.testing.assert_allclose(fused.observed_fraction, expected_fraction)


def test_window_stack_drops_geometry_if_fused_feature_count_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        genome=Genome(
            working_dim=8,
            temporal_memory_depth=2,
            temporal_fusion_mode=TemporalFusionMode.WINDOW_STACK,
        )
    )
    observation = ObservationPacket(data=np.ones(2), metadata=_metadata(2))

    def stack_with_extra_features(
        buffer: list[ObservationPacket],
    ) -> np.ndarray:
        return np.concatenate([sample.data for sample in buffer])

    monkeypatch.setattr(temporal, "_fuse_window_stack", stack_with_extra_features)
    apply_temporal_observation(agent, observation)
    fused = apply_temporal_observation(agent, observation)

    assert fused.data.size == 4
    assert fused.metadata is None
    assert fused.observed_fraction is None


def test_numeric_only_streams_remain_metadata_free() -> None:
    stream = Stream(
        stream_type=StreamType.RESIDUAL,
        dimensionality=2,
        current_data=np.ones(2),
    )
    assert stream.metadata is None
    assert stream.observation().metadata is None


def test_residual_republication_drops_source_geometry() -> None:
    world = World(SimulationConfig(initial_population=2, seed=7))
    agent = Agent(genome=Genome(working_dim=8))
    world._init_agent_model(agent)

    assert agent.state.output_stream_id is not None
    residual = world.streams[agent.state.output_stream_id]
    assert residual.stream_type == StreamType.RESIDUAL
    assert residual.metadata is None
    assert residual.current_status.size == 0
