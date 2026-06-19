"""Simulation configuration parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenePoolConfig(BaseModel):
    """Constraints on initial genome distributions and allowed trait values."""

    available_compression_types: list[str] | None = Field(
        default=None,
        description="Allowed compression types; None = all",
    )
    n_components_range: tuple[int, int] = Field(default=(1, 5))
    escalation_threshold_range: tuple[float, float] = Field(default=(0.3, 0.9))
    metabolic_efficiency_range: tuple[float, float] = Field(default=(0.5, 2.0))
    development_duration_range: tuple[int, int] = Field(default=(3, 10))
    working_dim_range: tuple[int, int] = Field(default=(8, 64))
    max_temporal_depth: int = Field(default=20, ge=0, le=100)
    n_blocks: int = Field(default=10, ge=1, description="Blocks for block_specialize sensing")
    available_sensing_strategies: list[str] | None = None
    available_temporal_modes: list[str] = Field(
        default_factory=lambda: ["none", "ema", "window_stack"]
    )
    available_spatial_strategies: list[str] | None = None
    available_residual_policies: list[str] | None = None
    available_escalation_modes: list[str] | None = None


class SimulationConfig(BaseModel):
    """Global configuration for a TattleTots simulation run."""

    max_population: int = Field(default=100, ge=2, description="Population cap")
    initial_population: int = Field(default=20, ge=2, description="Starting population size")
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    reproduction_energy_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of threshold energy passed to offspring",
    )
    subsidy_rate: float = Field(
        default=0.1,
        ge=0.0,
        description="Fraction of downstream yield passed upstream as subsidy",
    )
    initial_info_energy: float = Field(default=1.0, gt=0.0)
    initial_attn_energy: float = Field(default=1.0, gt=0.0)
    trust_delta_pos: float = Field(default=0.05, gt=0.0)
    trust_delta_neg: float = Field(default=0.2, gt=0.0)
    trust_delta_miss: float = Field(default=0.1, gt=0.0)
    false_alarm_penalty: float = Field(
        default=0.3, ge=0.0, description="Attention energy penalty for false alarms"
    )
    recombination_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability that reproduction is sexual vs asexual",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    max_steps: int = Field(default=1000, ge=1, description="Maximum simulation steps")
    max_stream_dim: int = Field(
        default=30,
        ge=1,
        description=(
            "Maximum dimensionality for combined inputs and residual streams. "
            "Prevents exponential vector growth through the trophic chain."
        ),
    )
    default_working_dim: int = Field(default=30, ge=8, le=256)
    max_working_dim: int = Field(default=256, ge=8, le=1024)
    extinction_check_window: int = Field(
        default=50, ge=1, description="Steps between stability checks"
    )
    use_gpu: bool = Field(
        default=False,
        description=(
            "Offload array math to the GPU via CuPy. Requires the [gpu] optional "
            "dependency (cupy-cuda12x). Falls back to NumPy silently if CuPy is "
            "not installed or no CUDA device is found."
        ),
    )
    # Compute complexity cost rates
    temporal_cost_rate: float = Field(default=0.001, ge=0.0)
    projection_cost_rate: float = Field(default=0.0005, ge=0.0)
    spatial_cost_rate: float = Field(default=0.02, ge=0.0)
    storage_cost_rate: float = Field(default=0.0001, ge=0.0)
    refine_cost_multiplier: float = Field(default=1.0, ge=0.0)
    escalation_cost_rate: float = Field(default=0.0002, ge=0.0)
    juvenile_maintenance_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maintenance cost multiplier for juveniles",
    )
    lineage_signature_tolerance: float = Field(
        default=0.5,
        ge=0.0,
        description="Max |parent_sig - child_sig| for lineage subsidy",
    )
    mimesis_learning_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    n_spatial_blocks: int = Field(
        default=10, ge=1, description="Uniform blocks for high-dim streams"
    )
