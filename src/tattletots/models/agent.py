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
