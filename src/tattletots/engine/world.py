"""World: the complete state of a TattleTots simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.attention import allocate_attention, compute_attention_income
from tattletots.engine.compression import CompressionModel, create_compression_model
from tattletots.engine.config import GenePoolConfig, SimulationConfig
from tattletots.engine.development import (
    apply_mimesis,
    apply_parental_investment,
    juvenile_maintenance_cost,
    lineage_subsidy_eligible,
    select_role_models,
)
from tattletots.engine.domestication import apply_shaping, compute_shaping_signal
from tattletots.engine.escalation import should_escalate
from tattletots.engine.reproduction import attempt_reproduction
from tattletots.engine.residual import apply_residual_policy
from tattletots.engine.sensing import gather_raw_stream_data, prepare_agent_input
from tattletots.engine.spatial import apply_spatial_mask, infer_spatial_location
from tattletots.engine.temporal import apply_temporal_fusion
from tattletots.engine.trophic import compute_trophic_level, select_input_streams
from tattletots.engine.trust import penalize_missed_events, verify_reports
from tattletots.engine.whistleblowing import (
    compute_dishonesty_score,
    create_output_stream,
    identify_whistleblower_targets,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome, ParentalStrategy, ResidualPolicy, SpatialStrategy
from tattletots.models.location import EventLocation
from tattletots.models.report import Report
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User
from tattletots.telemetry.recorder import StepRecord, TelemetryRecorder

LocationInferenceFn = Callable[[list[NDArray[np.float64]], list[str]], EventLocation]
DimToLocationFn = Callable[[int], EventLocation]


@dataclass
class World:
    """Complete simulation state: agents, streams, users, clock."""

    config: SimulationConfig
    gene_pool: GenePoolConfig | None = None
    agents: dict[str, Agent] = field(default_factory=dict)
    streams: dict[str, Stream] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    compression_models: dict[str, CompressionModel] = field(default_factory=dict)
    refine_models: dict[str, CompressionModel] = field(default_factory=dict)
    time_step: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    _active_locations: frozenset[EventLocation] = field(default_factory=frozenset)
    _ground_truth_active: bool = False
    _location_inference: LocationInferenceFn | None = None
    _dim_to_location: DimToLocationFn | None = None
    _ground_truth_vector: NDArray[np.float64] | None = None
    last_reports: list[Report] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)

    def add_stream(self, stream: Stream) -> None:
        """Register a data stream in the world."""
        self.streams[stream.id] = stream

    def add_user(self, user: User) -> None:
        """Register a human user."""
        self.users[user.id] = user

    def seed_population(self, genomes: list[Genome] | None = None) -> None:
        """Initialize the agent population from seed genomes or random generation."""
        n = self.config.initial_population

        if genomes is None:
            genomes = [self._random_genome() for _ in range(n)]

        for genome in genomes[:n]:
            agent = Agent(
                genome=genome,
                state=AgentState(
                    energy=EnergyReserves(
                        information=self.config.initial_info_energy,
                        attention=self.config.initial_attn_energy,
                    ),
                    lifecycle=LifecycleStage.ADULT,
                    effective_escalation_threshold=genome.escalation_threshold,
                ),
            )
            self.agents[agent.id] = agent
            self._init_agent_model(agent)

    def set_event_state(self, active_locations: list[EventLocation]) -> None:
        """Set active event locations for report verification this step."""
        self._active_locations = frozenset(active_locations)
        self._ground_truth_active = len(self._active_locations) > 0

    def set_ground_truth(self, active: bool) -> None:
        """Legacy wrapper: active event with unknown location uses empty location set."""
        if active:
            raise ValueError(
                "set_ground_truth(True) is ambiguous without locations; "
                "use set_event_state(adapter.get_active_locations(step)) instead"
            )
        self.set_event_state([])

    def set_location_inference(self, fn: LocationInferenceFn) -> None:
        """Register domain callback to infer report location from agent input streams."""
        self._location_inference = fn

    def set_dim_to_location(self, fn: DimToLocationFn) -> None:
        """Register mapping from dimension index to spatial location."""
        self._dim_to_location = fn

    def set_ground_truth_vector(self, vec: NDArray[np.float64]) -> None:
        """Optional ground truth vector for whistleblowing."""
        self._ground_truth_vector = vec

    def step(self) -> StepRecord:
        """Execute one simulation time step."""
        self.time_step += 1
        reports: list[Report] = []
        births: list[str] = []
        deaths: list[str] = []

        living_agents = [a for a in self.agents.values() if a.is_alive]

        # 1. Trophic attachment
        available_streams = list(self.streams.values())
        for agent in living_agents:
            new_inputs = select_input_streams(agent, available_streams, max_inputs=3, rng=self.rng)
            agent.state.input_stream_ids = new_inputs

        # 2. Development / mimesis (juveniles)
        for agent in living_agents:
            if agent.state.lifecycle == LifecycleStage.JUVENILE:
                models = select_role_models(agent, self.agents, self.users)
                apply_mimesis(agent, models, self.config)

        # 3. Sensing → temporal → spatial → compression → residual
        for agent in living_agents:
            self._compress(agent)

        # 4. Escalation (adults only)
        for agent in living_agents:
            if agent.state.lifecycle != LifecycleStage.ADULT:
                continue
            report = self._maybe_escalate(agent)
            if report is not None:
                reports.append(report)
                self._publish_output_stream(agent)

        # 5. Whistleblowing checks
        self._process_whistleblowing(living_agents)

        # 6. Trust verification
        verified_reports = verify_reports(reports, self._active_locations, self.users, self.config)
        self.last_reports = verified_reports

        missed: list[str] = []
        if self._active_locations:
            reports_by_agent: dict[str, list[Report]] = {}
            for report in reports:
                reports_by_agent.setdefault(report.agent_id, []).append(report)

            for agent in living_agents:
                if agent.state.lifecycle != LifecycleStage.ADULT:
                    continue
                agent_reports = reports_by_agent.get(agent.id, [])
                if not agent_reports:
                    missed.append(agent.id)
                    continue
                if not any(r.location in self._active_locations for r in agent_reports):
                    missed.append(agent.id)
            penalize_missed_events(missed, self.users, self.config)

        # 7. Attention allocation
        all_allocations: dict[str, dict[str, float]] = {}
        for user in self.users.values():
            all_allocations[user.id] = allocate_attention(
                user, living_agents, use_gpu=self.config.use_gpu
            )

        # 8. Energy accounting
        for agent in living_agents:
            self._apply_energy(agent, all_allocations, verified_reports)

        # 9. Domestication
        self._apply_domestication(living_agents)

        # 10. Death check
        for agent in living_agents:
            if not agent.state.energy.is_alive:
                agent.kill()
                deaths.append(agent.id)
                if agent.state.output_stream_id:
                    self.streams.pop(agent.state.output_stream_id, None)
                if agent.state.output_claim_stream_id:
                    self.streams.pop(agent.state.output_claim_stream_id, None)
                if agent.state.curated_stream_id:
                    self.streams.pop(agent.state.curated_stream_id, None)
                self.compression_models.pop(agent.id, None)
                self.refine_models.pop(agent.id, None)

        # 11. Reproduction
        still_alive = [a for a in self.agents.values() if a.is_alive]
        offspring = attempt_reproduction(still_alive, self.config, self.rng)
        for child in offspring:
            self.agents[child.id] = child
            self._init_agent_model(child)
            self._apply_parental_effects(child)
            births.append(child.id)

        # 12. Age advancement
        for agent in self.agents.values():
            if agent.is_alive:
                agent.advance_age()

        # Cleanup orphaned streams
        living_ids = {a.id for a in self.agents.values() if a.is_alive}
        orphaned = [
            sid
            for sid, s in self.streams.items()
            if s.source_agent_id is not None and s.source_agent_id not in living_ids
        ]
        for sid in orphaned:
            del self.streams[sid]

        record = self._build_step_record(
            reports=verified_reports,
            births=births,
            deaths=deaths,
            missed=missed,
        )
        self.telemetry.record_step(record)
        return record

    def run(self, steps: int | None = None) -> list[StepRecord]:
        """Run the simulation for the specified number of steps."""
        n = steps or self.config.max_steps
        records: list[StepRecord] = []
        for _ in range(n):
            record = self.step()
            records.append(record)
            if not any(a.is_alive for a in self.agents.values()):
                break
        return records

    @property
    def living_population(self) -> int:
        return sum(1 for a in self.agents.values() if a.is_alive)

    @property
    def trophic_levels(self) -> dict[str, float]:
        agent_inputs: dict[str, list[str]] = {}
        stream_sources: dict[str, str | None] = {}

        for agent in self.agents.values():
            if agent.is_alive:
                agent_inputs[agent.id] = agent.state.input_stream_ids

        for stream in self.streams.values():
            stream_sources[stream.id] = stream.source_agent_id

        levels: dict[str, float] = {}
        memo: dict[str, float] = {}
        for agent_id in agent_inputs:
            levels[agent_id] = compute_trophic_level(agent_id, agent_inputs, stream_sources, memo)
        return levels

    def _random_genome(self) -> Genome:
        return Genome.random_genome(
            self.rng,
            n_streams=max(len(self.streams), 1),
            n_users=max(len(self.users), 1),
            gene_pool=self.gene_pool,
        )

    def _init_agent_model(self, agent: Agent) -> None:
        window = (
            max(20, agent.genome.temporal_memory_depth)
            if agent.genome.temporal_memory_depth > 0
            else 20
        )
        model = create_compression_model(
            agent.genome.compression_type,
            agent.genome.n_components,
            agent.genome.metabolic_efficiency,
            use_gpu=self.config.use_gpu,
            window_size=window,
        )
        self.compression_models[agent.id] = model

        if agent.genome.residual_policy == ResidualPolicy.REFINE:
            self.refine_models[agent.id] = create_compression_model(
                agent.genome.compression_type,
                agent.genome.n_components,
                agent.genome.metabolic_efficiency,
                use_gpu=self.config.use_gpu,
            )

        out_dim = agent.genome.working_dim
        residual_stream = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=out_dim,
            source_agent_id=agent.id,
            label=f"residual_{agent.id[:8]}",
        )
        self.streams[residual_stream.id] = residual_stream
        agent.state.output_stream_id = residual_stream.id

    def _prepare_input_pipeline(self, agent: Agent) -> NDArray[np.float64]:
        """Run sensing → temporal → spatial pipeline."""
        sensed, _ = prepare_agent_input(agent, self.streams, self.config)
        if sensed.size == 0:
            return sensed
        temporal = apply_temporal_fusion(agent, sensed)
        spatial = apply_spatial_mask(
            agent,
            temporal,
            n_blocks=self.config.n_spatial_blocks,
            dim_to_location=self._dim_to_location,
        )
        agent.state.projected_input = spatial
        return spatial

    def _compress(self, agent: Agent) -> None:
        """Run full compression pipeline on agent inputs."""
        model = self.compression_models.get(agent.id)
        if model is None:
            return

        combined = self._prepare_input_pipeline(agent)
        if combined.size == 0:
            agent.state.last_step_yield = 0.0
            return

        max_dim = min(agent.genome.working_dim, self.config.max_stream_dim)
        if combined.size > max_dim:
            combined = combined[:max_dim]

        residual, info_yield = model.fit_transform(combined)

        refine_model = self.refine_models.get(agent.id)
        output, adjusted_yield, out_dim = apply_residual_policy(
            agent,
            residual,
            info_yield,
            refine_model=refine_model,
            max_dim=max_dim,
        )

        agent.state.signal_vector = model.get_signal_vector()
        agent.state.last_step_yield = adjusted_yield
        agent.state.cumulative_yield += adjusted_yield

        compute_cost = agent.genome.total_compute_cost(
            self.config,
            residual_buffer_len=len(agent.state.residual_buffer),
            residual_buffer_dim=max_dim,
        )
        agent.state.last_compute_cost_paid = compute_cost

        if agent.state.output_stream_id:
            out_stream = self.streams.get(agent.state.output_stream_id)
            if out_stream is not None:
                if out_dim != out_stream.dimensionality:
                    out_stream.dimensionality = out_dim
                out_stream.update(output)

    def _maybe_escalate(self, agent: Agent) -> Report | None:
        """Agent decides whether to escalate based on anomaly score vs threshold."""
        model = self.compression_models.get(agent.id)
        if model is None:
            return None

        combined = agent.state.projected_input
        if combined.size == 0:
            combined = self._prepare_input_pipeline(agent)
        if combined.size == 0:
            return None

        anomaly, threshold, fire = should_escalate(agent, model, combined)
        if not fire:
            return None

        user_ids = list(self.users.keys())
        if not user_ids:
            return None

        affinity = agent.genome.target_user_affinity
        if affinity.size >= len(user_ids):
            target_idx = int(np.argmax(affinity[: len(user_ids)]))
        else:
            target_idx = int(self.rng.integers(0, len(user_ids)))

        agent.state.reports_issued += 1

        location = infer_spatial_location(
            agent,
            combined,
            n_blocks=self.config.n_spatial_blocks,
            dim_to_location=self._dim_to_location,
        )
        if (
            agent.genome.spatial_strategy == SpatialStrategy.GLOBAL
            and self._location_inference is not None
        ):
            raw_data, raw_labels = gather_raw_stream_data(agent, self.streams)
            if raw_data:
                location = self._location_inference(raw_data, raw_labels)

        return Report(
            agent_id=agent.id,
            target_user_id=user_ids[target_idx],
            time_step=self.time_step,
            signal_vector=agent.state.signal_vector,
            confidence=min(1.0, anomaly / (threshold + 1e-10)),
            anomaly_score=anomaly,
            location=location,
        )

    def _publish_output_stream(self, agent: Agent) -> None:
        """Publish OUTPUT stream on escalation for whistleblower consumption."""
        signal = agent.state.signal_vector
        if signal.size == 0:
            return
        if agent.state.output_claim_stream_id is None:
            out = create_output_stream(agent, signal, signal.size)
            self.streams[out.id] = out
            agent.state.output_claim_stream_id = out.id
        else:
            stream = self.streams.get(agent.state.output_claim_stream_id)
            if stream is not None:
                if stream.dimensionality != signal.size:
                    stream.dimensionality = signal.size
                stream.update(signal)

    def _process_whistleblowing(self, living_agents: list[Agent]) -> None:
        """Agents consuming OUTPUT streams detect dishonesty."""
        if self._ground_truth_vector is None:
            return
        gt = self._ground_truth_vector
        for agent in living_agents:
            targets = identify_whistleblower_targets(agent, self.streams)
            if not targets:
                continue
            for stream_id in targets:
                stream = self.streams.get(stream_id)
                if stream is None or stream.current_data.size == 0:
                    continue
                score = compute_dishonesty_score(stream.current_data, gt)
                if score > 0.5:
                    agent.state.energy.apply_attention_delta(0.05)

    def _apply_parental_effects(self, child: Agent) -> None:
        """Apply parental strategy after child creation."""
        for pid in child.state.parent_ids:
            parent = self.agents.get(pid)
            if parent is None or not parent.is_alive:
                continue
            if not lineage_subsidy_eligible(parent, child, self.config):
                continue
            apply_parental_investment(parent, child, self.config)
            if parent.genome.parental_strategy == ParentalStrategy.MARSUPIAL:
                self._setup_marsupial_stream(parent, child)

    def _setup_marsupial_stream(self, parent: Agent, child: Agent) -> None:
        """Parent routes curated residual sub-stream to juvenile."""
        if parent.state.output_stream_id is None:
            return
        parent_residual = self.streams.get(parent.state.output_stream_id)
        if parent_residual is None or parent_residual.current_data.size == 0:
            return
        dim = min(parent_residual.dimensionality, child.genome.working_dim)
        curated = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=dim,
            source_agent_id=parent.id,
            label=f"curated_{child.id[:8]}",
            current_data=parent_residual.current_data[:dim].copy(),
        )
        self.streams[curated.id] = curated
        child.state.curated_stream_id = curated.id
        if child.state.lifecycle == LifecycleStage.JUVENILE:
            child.state.input_stream_ids = [curated.id]

    def _apply_energy(
        self,
        agent: Agent,
        allocations: dict[str, dict[str, float]],
        reports: list[Report],
    ) -> None:
        """Apply dual-currency energy changes for one agent."""
        compute_cost = agent.state.last_compute_cost_paid or agent.genome.total_compute_cost(
            self.config,
            residual_buffer_len=len(agent.state.residual_buffer),
            residual_buffer_dim=agent.genome.working_dim,
        )

        info_delta = -compute_cost + agent.state.last_step_yield

        if agent.state.output_stream_id:
            downstream_count = sum(
                1
                for a in self.agents.values()
                if a.is_alive and agent.state.output_stream_id in a.state.input_stream_ids
            )
            info_delta += downstream_count * self.config.subsidy_rate

        agent.state.energy.apply_info_delta(info_delta)

        attn_income = compute_attention_income(agent, list(self.users.values()), allocations)
        agent.state.cumulative_attention += attn_income
        maint = juvenile_maintenance_cost(agent, self.config)
        attn_delta = attn_income - maint

        agent_reports = [r for r in reports if r.agent_id == agent.id]
        false_alarms = sum(1 for r in agent_reports if r.verified and not r.correct)
        attn_delta -= false_alarms * self.config.false_alarm_penalty
        agent.state.false_alarms += false_alarms
        agent.state.correct_reports += sum(1 for r in agent_reports if r.verified and r.correct)

        agent.state.energy.apply_attention_delta(attn_delta)

    def _apply_domestication(self, living_agents: list[Agent]) -> None:
        for downstream in living_agents:
            for stream_id in downstream.state.input_stream_ids:
                stream = self.streams.get(stream_id)
                if stream is None or stream.stream_type != StreamType.RESIDUAL:
                    continue
                upstream_id = stream.source_agent_id
                if upstream_id is None:
                    continue
                upstream = self.agents.get(upstream_id)
                if upstream is None or not upstream.is_alive:
                    continue
                signal = compute_shaping_signal(downstream, upstream)
                if signal.size > 0:
                    apply_shaping(upstream, [signal])

    def _build_step_record(
        self,
        reports: list[Report],
        births: list[str],
        deaths: list[str],
        missed: list[str],
    ) -> StepRecord:
        living = [a for a in self.agents.values() if a.is_alive]
        juveniles = [a for a in living if a.state.lifecycle == LifecycleStage.JUVENILE]
        adults = [a for a in living if a.state.lifecycle == LifecycleStage.ADULT]
        return StepRecord(
            time_step=self.time_step,
            population=len(living),
            births=len(births),
            deaths=len(deaths),
            reports_issued=len(reports),
            correct_reports=sum(1 for r in reports if r.correct),
            false_alarms=sum(1 for r in reports if r.verified and not r.correct),
            mean_info_energy=(
                float(np.mean([a.state.energy.information for a in living])) if living else 0.0
            ),
            mean_attn_energy=(
                float(np.mean([a.state.energy.attention for a in living])) if living else 0.0
            ),
            max_trophic_level=max(self.trophic_levels.values(), default=1.0),
            n_streams=len(self.streams),
            ground_truth_active=self._ground_truth_active,
            active_location_count=len(self._active_locations),
            total_info_yield=sum(a.state.last_step_yield for a in living),
            total_attn_income=sum(a.state.energy.attention for a in living),
            total_compute_cost=sum(a.state.last_compute_cost_paid for a in living),
            total_maintenance_cost=sum(juvenile_maintenance_cost(a, self.config) for a in living),
            n_juveniles=len(juveniles),
            n_adults=len(adults),
            mean_generation=(
                float(np.mean([a.state.generation for a in living])) if living else 0.0
            ),
            n_compression_types=len({a.genome.compression_type for a in living}),
            missed_events=len(missed),
            mean_working_dim=(
                float(np.mean([a.genome.working_dim for a in living])) if living else 0.0
            ),
            mean_memory_depth=(
                float(np.mean([a.genome.temporal_memory_depth for a in living])) if living else 0.0
            ),
            n_sensing_strategies=len({a.genome.sensing_strategy for a in living}),
            n_residual_policies=len({a.genome.residual_policy for a in living}),
        )
