"""World: the complete state of a TattleTots simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tattletots.engine.attention import allocate_attention, compute_attention_income
from tattletots.engine.compression import CompressionModel, create_compression_model
from tattletots.engine.config import SimulationConfig
from tattletots.engine.domestication import apply_shaping, compute_shaping_signal
from tattletots.engine.reproduction import attempt_reproduction
from tattletots.engine.trophic import compute_trophic_level, select_input_streams
from tattletots.engine.trust import penalize_missed_events, verify_reports
from tattletots.engine.whistleblowing import (  # noqa: F401
    compute_dishonesty_score,
    create_output_stream,
)
from tattletots.models.agent import Agent, AgentState, LifecycleStage
from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome
from tattletots.models.report import Report
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User
from tattletots.telemetry.recorder import StepRecord, TelemetryRecorder


@dataclass
class World:
    """Complete simulation state: agents, streams, users, clock."""

    config: SimulationConfig
    agents: dict[str, Agent] = field(default_factory=dict)
    streams: dict[str, Stream] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    compression_models: dict[str, CompressionModel] = field(default_factory=dict)
    time_step: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    _ground_truth_active: bool = False

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
                ),
            )
            self.agents[agent.id] = agent
            self._init_agent_model(agent)

    def set_ground_truth(self, active: bool) -> None:
        """Set whether a true event is occurring (for verification)."""
        self._ground_truth_active = active

    def step(self) -> StepRecord:
        """Execute one simulation time step.

        Order of operations:
        1. Agents select input streams (trophic attachment)
        2. Agents compress their inputs (metabolism)
        3. Agents decide whether to escalate (reporting)
        4. Reports are verified against ground truth (trust update)
        5. Users allocate attention (attention economics)
        6. Energy accounting (dual-currency bookkeeping)
        7. Domestication signals flow downstream → upstream
        8. Death check (starvation)
        9. Reproduction (if above threshold)
        10. Age advancement
        """
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

        # 2. Compression (metabolism)
        for agent in living_agents:
            self._compress(agent)

        # 3. Escalation decision
        for agent in living_agents:
            if agent.state.lifecycle != LifecycleStage.ADULT:
                continue
            report = self._maybe_escalate(agent)
            if report is not None:
                reports.append(report)

        # 4. Trust verification
        verified_reports = verify_reports(
            reports, self._ground_truth_active, self.users, self.config
        )

        # Penalize agents that missed a true event
        if self._ground_truth_active:
            reporting_ids = {r.agent_id for r in reports}
            missed = [
                a.id
                for a in living_agents
                if a.id not in reporting_ids and a.state.lifecycle == LifecycleStage.ADULT
            ]
            penalize_missed_events(missed, self.users, self.config)

        # 5. Attention allocation
        all_allocations: dict[str, dict[str, float]] = {}
        for user in self.users.values():
            all_allocations[user.id] = allocate_attention(user, living_agents)

        # 6. Energy accounting
        for agent in living_agents:
            self._apply_energy(agent, all_allocations, verified_reports)

        # 7. Domestication
        self._apply_domestication(living_agents)

        # 8. Death check
        for agent in living_agents:
            if not agent.state.energy.is_alive:
                agent.kill()
                deaths.append(agent.id)
                # Remove agent's residual stream and compression model
                if agent.state.output_stream_id:
                    self.streams.pop(agent.state.output_stream_id, None)
                self.compression_models.pop(agent.id, None)

        # 9. Reproduction
        still_alive = [a for a in self.agents.values() if a.is_alive]
        offspring = attempt_reproduction(still_alive, self.config, self.rng)
        for child in offspring:
            self.agents[child.id] = child
            self._init_agent_model(child)
            births.append(child.id)

        # 10. Age advancement
        for agent in self.agents.values():
            if agent.is_alive:
                agent.advance_age()

        # Cleanup orphaned streams (residuals from dead agents)
        living_ids = {a.id for a in self.agents.values() if a.is_alive}
        orphaned = [
            sid
            for sid, s in self.streams.items()
            if s.source_agent_id is not None and s.source_agent_id not in living_ids
        ]
        for sid in orphaned:
            del self.streams[sid]

        # Record telemetry
        record = self._build_step_record(reports=verified_reports, births=births, deaths=deaths)
        self.telemetry.record_step(record)
        return record

    def run(self, steps: int | None = None) -> list[StepRecord]:
        """Run the simulation for the specified number of steps."""
        n = steps or self.config.max_steps
        records: list[StepRecord] = []
        for _ in range(n):
            record = self.step()
            records.append(record)
            # Check for total extinction
            if not any(a.is_alive for a in self.agents.values()):
                break
        return records

    @property
    def living_population(self) -> int:
        """Count of currently alive agents."""
        return sum(1 for a in self.agents.values() if a.is_alive)

    @property
    def trophic_levels(self) -> dict[str, float]:
        """Compute trophic level for every living agent."""
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

    # --- Private helpers ---

    def _random_genome(self) -> Genome:
        """Generate a random genome for population seeding."""
        from tattletots.models.genome import CompressionType

        n_streams = max(len(self.streams), 1)
        n_users = max(len(self.users), 1)

        return Genome(
            compression_type=list(CompressionType)[int(self.rng.integers(0, len(CompressionType)))],
            n_components=int(self.rng.integers(1, 6)),
            input_preference=self.rng.dirichlet(np.ones(n_streams)),
            escalation_threshold=float(self.rng.uniform(0.3, 0.9)),
            target_user_affinity=self.rng.dirichlet(np.ones(n_users)),
            metabolic_efficiency=float(self.rng.uniform(0.5, 2.0)),
            compute_cost=float(self.rng.uniform(0.05, 0.2)),
            maintenance_cost=float(self.rng.uniform(0.02, 0.1)),
            reproduction_threshold=float(self.rng.uniform(1.5, 3.0)),
            domestication_sensitivity=float(self.rng.uniform(0.0, 0.3)),
        )

    def _init_agent_model(self, agent: Agent) -> None:
        """Initialize compression model for an agent."""
        model = create_compression_model(
            agent.genome.compression_type,
            agent.genome.n_components,
            agent.genome.metabolic_efficiency,
        )
        self.compression_models[agent.id] = model

        # Create residual output stream
        dim = max(
            (self.streams[sid].dimensionality for sid in agent.state.input_stream_ids),
            default=10,
        )
        residual_stream = Stream(
            stream_type=StreamType.RESIDUAL,
            dimensionality=dim,
            source_agent_id=agent.id,
            label=f"residual_{agent.id[:8]}",
        )
        self.streams[residual_stream.id] = residual_stream
        agent.state.output_stream_id = residual_stream.id

    def _compress(self, agent: Agent) -> None:
        """Run compression model on agent's inputs."""
        model = self.compression_models.get(agent.id)
        if model is None:
            return

        # Gather input data
        input_data: list[np.ndarray] = []
        for stream_id in agent.state.input_stream_ids:
            stream = self.streams.get(stream_id)
            if stream is not None and stream.current_data.size > 0:
                input_data.append(stream.current_data)

        if not input_data:
            agent.state.last_step_yield = 0.0
            return

        combined = np.concatenate(input_data)
        # Cap input dimensionality to prevent exponential growth through trophic chain
        max_dim = 30
        if combined.size > max_dim:
            combined = combined[:max_dim]

        residual, info_yield = model.fit_transform(combined)

        # Update agent state
        agent.state.signal_vector = model.get_signal_vector()
        agent.state.last_step_yield = info_yield
        agent.state.cumulative_yield += info_yield

        # Update residual stream (capped to max_dim)
        if agent.state.output_stream_id:
            out_stream = self.streams.get(agent.state.output_stream_id)
            if out_stream is not None:
                capped_residual = residual[:max_dim]
                if capped_residual.size != out_stream.dimensionality:
                    out_stream.dimensionality = capped_residual.size
                out_stream.update(capped_residual)

    def _maybe_escalate(self, agent: Agent) -> Report | None:
        """Agent decides whether to escalate based on anomaly score vs threshold."""
        model = self.compression_models.get(agent.id)
        if model is None:
            return None

        # Get current input for anomaly scoring
        input_data: list[np.ndarray] = []
        for stream_id in agent.state.input_stream_ids:
            stream = self.streams.get(stream_id)
            if stream is not None and stream.current_data.size > 0:
                input_data.append(stream.current_data)

        if not input_data:
            return None

        combined = np.concatenate(input_data)
        # Cap input dimensionality (must match cap used in _compress)
        max_dim = 30
        if combined.size > max_dim:
            combined = combined[:max_dim]
        anomaly = model.anomaly_score(combined)

        if anomaly < agent.genome.escalation_threshold:
            return None

        # Pick target user
        user_ids = list(self.users.keys())
        if not user_ids:
            return None

        affinity = agent.genome.target_user_affinity
        if affinity.size >= len(user_ids):
            target_idx = int(np.argmax(affinity[: len(user_ids)]))
        else:
            target_idx = int(self.rng.integers(0, len(user_ids)))

        agent.state.reports_issued += 1
        return Report(
            agent_id=agent.id,
            target_user_id=user_ids[target_idx],
            time_step=self.time_step,
            signal_vector=agent.state.signal_vector,
            confidence=min(1.0, anomaly / (agent.genome.escalation_threshold + 1e-10)),
            anomaly_score=anomaly,
        )

    def _apply_energy(
        self,
        agent: Agent,
        allocations: dict[str, dict[str, float]],
        reports: list[Report],
    ) -> None:
        """Apply dual-currency energy changes for one agent."""
        model = self.compression_models.get(agent.id)

        # Information energy: yield - compute_cost + subsidy
        info_delta = -agent.genome.compute_cost
        if model is not None:
            # Use this step's compression yield
            info_delta += agent.state.last_step_yield
        # Subsidy from downstream agents consuming this agent's residual
        if agent.state.output_stream_id:
            downstream_count = sum(
                1
                for a in self.agents.values()
                if a.is_alive and agent.state.output_stream_id in a.state.input_stream_ids
            )
            info_delta += downstream_count * self.config.subsidy_rate

        agent.state.energy.apply_info_delta(info_delta)

        # Attention energy: income - maintenance - false_alarm_penalty
        attn_income = compute_attention_income(agent, list(self.users.values()), allocations)
        agent.state.cumulative_attention += attn_income
        attn_delta = attn_income - agent.genome.maintenance_cost

        # False alarm penalty
        agent_reports = [r for r in reports if r.agent_id == agent.id]
        false_alarms = sum(1 for r in agent_reports if r.verified and not r.correct)
        attn_delta -= false_alarms * self.config.false_alarm_penalty
        agent.state.false_alarms += false_alarms
        agent.state.correct_reports += sum(1 for r in agent_reports if r.verified and r.correct)

        agent.state.energy.apply_attention_delta(attn_delta)

    def _apply_domestication(self, living_agents: list[Agent]) -> None:
        """Apply domestication signals from downstream to upstream agents."""
        # Build map of who consumes whose residual
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
    ) -> StepRecord:
        """Build telemetry record for this step."""
        living = [a for a in self.agents.values() if a.is_alive]
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
        )
