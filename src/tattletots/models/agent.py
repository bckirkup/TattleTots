"""Agent model: the core replicating entity in the TattleTots ecology."""

from __future__ import annotations

import enum
import uuid

import numpy as np
from pydantic import BaseModel, Field

from tattletots.models.energy import EnergyReserves
from tattletots.models.genome import Genome


class LifecycleStage(enum.StrEnum):
    """Agent lifecycle stages."""

    JUVENILE = "juvenile"
    ADULT = "adult"
    DEAD = "dead"


class AgentState(BaseModel):
    """Mutable runtime state for an agent (separate from heritable genome)."""

    model_config = {"arbitrary_types_allowed": True}

    energy: EnergyReserves = Field(default_factory=EnergyReserves)
    lifecycle: LifecycleStage = Field(default=LifecycleStage.JUVENILE)
    age: int = Field(default=0, ge=0, description="Time steps since birth")
    input_stream_ids: list[str] = Field(
        default_factory=list,
        description="IDs of streams this agent currently consumes",
    )
    output_stream_id: str | None = Field(
        default=None, description="ID of this agent's residual output stream"
    )
    compression_state: dict[str, float] = Field(
        default_factory=dict,
        description="Internal state of the compression model (model-specific)",
    )
    cumulative_yield: float = Field(default=0.0, description="Lifetime information yield")
    last_step_yield: float = Field(default=0.0, description="Info yield from most recent step")
    cumulative_attention: float = Field(default=0.0, description="Lifetime attention income")
    reports_issued: int = Field(default=0, ge=0)
    correct_reports: int = Field(default=0, ge=0)
    false_alarms: int = Field(default=0, ge=0)
    parent_ids: list[str] = Field(
        default_factory=list, description="Parent agent IDs (1 for asexual, 2 for sexual)"
    )
    generation: int = Field(default=0, ge=0)
    signal_vector: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Current signal vector (compressed representation of input)",
    )
    input_preference_override: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Runtime override of genome input_preference (set by domestication)",
    )
    anomaly_history: list[float] = Field(
        default_factory=list,
        description="Rolling window of raw anomaly scores for baseline normalization",
    )
    temporal_buffer: list[np.ndarray] = Field(
        default_factory=list,
        description="Ring buffer of post-sensing vectors for temporal fusion",
    )
    residual_buffer: list[np.ndarray] = Field(
        default_factory=list,
        description="Stored residuals awaiting emission (STORE policy)",
    )
    fusion_weights_override: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Runtime override of fusion_weights (mimesis/domestication)",
    )
    effective_escalation_threshold: float = Field(
        default=0.7,
        description="Computed threshold after adaptive calibration",
    )
    last_compute_cost_paid: float = Field(default=0.0)
    output_claim_stream_id: str | None = Field(
        default=None,
        description="OUTPUT stream published on escalation (whistleblowing)",
    )
    curated_stream_id: str | None = Field(
        default=None,
        description="Marsupial parent-curated stream for juvenile training",
    )
    last_spatial_mask: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Last spatial weight mask applied to input",
    )
    projected_input: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Last sensing+temporal+spatial processed input",
    )
    peer_trust: dict[str, float] = Field(
        default_factory=dict,
        description="Agent-to-agent trust, values in [0, 1]",
    )
    last_inferred_location: tuple[int, int] | None = Field(
        default=None,
        description="Where this agent observed signal this step",
    )
    last_anomaly_score: float = Field(default=0.0, ge=0.0)
    last_escalated: bool = Field(default=False)
    last_published_output: bool = Field(default=False)
    last_whistleblower_reports_issued: int = Field(default=0, ge=0)
    last_step_attention_income: float = Field(
        default=0.0,
        ge=0.0,
        description="Attention income this step (observable resourcing reward)",
    )
    last_step_info_subsidy: float = Field(
        default=0.0,
        ge=0.0,
        description="Downstream information subsidy received this step (observable)",
    )
    last_observed_dispatch: bool = Field(
        default=False,
        description="Whether this agent's escalation was linked to a dispatch this cycle",
    )
    last_observed_outcome_necessary: bool | None = Field(
        default=None,
        description="Observed post-dispatch necessity judgment when dispatched",
    )


class Agent(BaseModel):
    """A Tot: an information-processing agent in the ecology.

    Agents consume data streams, compress them, produce residuals, compete
    for human attention, and evolve over generations. They maintain dual
    energy reserves (information + attention) and die if either is depleted.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    genome: Genome = Field(default_factory=Genome)
    state: AgentState = Field(default_factory=AgentState)

    @property
    def is_alive(self) -> bool:
        """Check if agent is still viable."""
        return self.state.lifecycle != LifecycleStage.DEAD and self.state.energy.is_alive

    @property
    def can_reproduce(self) -> bool:
        """Check if agent has enough energy to reproduce."""
        return (
            self.state.lifecycle == LifecycleStage.ADULT
            and self.state.energy.total >= self.genome.reproduction_threshold
        )

    def advance_age(self) -> None:
        """Advance agent age by one step; transition juvenile → adult if mature."""
        self.state.age += 1
        if (
            self.state.lifecycle == LifecycleStage.JUVENILE
            and self.state.age >= self.genome.development_duration
        ):
            self.state.lifecycle = LifecycleStage.ADULT

    def kill(self) -> None:
        """Mark agent as dead."""
        self.state.lifecycle = LifecycleStage.DEAD

    def get_peer_trust(self, agent_id: str) -> float:
        return self.state.peer_trust.get(agent_id, 0.5)

    def update_peer_trust(
        self,
        agent_id: str,
        *,
        positive: bool = False,
        negative: bool = False,
        missed: bool = False,
        delta_pos: float = 0.05,
        delta_neg: float = 0.15,
        delta_miss: float = 0.1,
    ) -> None:
        current = self.get_peer_trust(agent_id)
        if positive:
            self.state.peer_trust[agent_id] = min(1.0, current + delta_pos)
        elif negative:
            self.state.peer_trust[agent_id] = max(0.0, current - delta_neg)
        elif missed:
            self.state.peer_trust[agent_id] = max(0.0, current - delta_miss)

    def spawn_offspring(self, rng: np.random.Generator, mutation_rate: float = 0.1) -> Agent:
        """Asexual reproduction: create a mutated offspring."""
        child_genome = self.genome.mutate(rng, rate=mutation_rate)
        # Parent pays energy cost
        cost = self.genome.reproduction_threshold / 2
        self.state.energy.information -= cost / 2
        self.state.energy.attention -= cost / 2

        return Agent(
            genome=child_genome,
            state=AgentState(
                energy=EnergyReserves(information=cost / 2, attention=cost / 2),
                parent_ids=[self.id],
                generation=self.state.generation + 1,
            ),
        )
