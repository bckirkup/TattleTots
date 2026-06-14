"""Simulation configuration parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
