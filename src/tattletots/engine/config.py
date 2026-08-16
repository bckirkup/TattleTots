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
    reproduction_coupling_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Strength of Liebig co-limitation by information and attention. "
            "Set to 0.0 for the legacy total-energy reproduction behavior. "
            "The legacy-equivalence setting uses both requirement scales at 1.0."
        ),
    )
    reproduction_information_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Global scale for genome information requirement stoichiometry",
    )
    reproduction_attention_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Global scale for genome attention requirement stoichiometry",
    )
    grounding_quality_strength: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description=(
            "Discount applied to ungrounded information yield. "
            "Set to 0.0 for the legacy yield behavior."
        ),
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
    correct_report_attention_value: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Extra attention value per verified-correct report, applied as the "
            "per-user value term v in income = alpha * v. At 0.0 (default) attention "
            "income is paid for trust-weighted relevance alone, independent of whether "
            "the agent was right."
        ),
    )
    false_alarm_break_even_precision: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Target break-even precision for reporting. When set, the per-false-alarm "
            "attention penalty is priced as value_per_correct_report * p / (1 - p) "
            "instead of the flat false_alarm_penalty, so an agent reporting above this "
            "precision gains attention by reporting and one below it loses attention. "
            "Requires correct_report_attention_value > 0, which supplies the value of a "
            "correct report; without it the flat penalty is used. At None (default) the "
            "penalty is flat, independent of the precision the instrument makes reachable."
        ),
    )
    reproduction_merit_ordering: bool = Field(
        default=False,
        description=(
            "Order eligible parents by reproductive co-limitation before the population "
            "cap is applied, so a binding cap rations opportunities by reserves rather "
            "than by agent creation order. At False (default) ordering is creation order."
        ),
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
    input_preference_slots: int = Field(
        default=32,
        ge=1,
        description="Number of genome slots used to hash stream attachment preferences.",
    )
    max_input_streams: int = Field(
        default=3,
        ge=1,
        description="Number of input streams each agent attaches to per step.",
    )
    grounded_input_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of an agent's input slots reserved for grounded raw streams. "
            "At 0.0 (default) attachment is unreserved and identical to legacy behavior."
        ),
    )
    grounded_attractiveness_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Attractiveness multiplier applied to grounded raw streams during "
            "attachment. At 1.0 (default) raw and residual streams compete unweighted."
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
    require_grounded_report_locations: bool = Field(
        default=False,
        description=(
            "Suppress reports when the agent has no coordinate-bearing evidence "
            "for the current step."
        ),
    )
    initiation_min_grounded_yield_share: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum grounded information-yield share before initiation is degenerate",
    )
    initiation_attention_insolvency_steps_fraction: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of steps that must be majority-insolvent for initiation degeneracy",
    )
    initiation_min_solvent_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Per-step solvent fraction below which a step is insolvent",
    )
    initiation_population_capacity_overshoot_factor: float = Field(
        default=1.0,
        ge=1.0,
        description="Peak-population multiple above mean attention capacity considered overshoot",
    )
    # Common Operating Picture
    cop_dispatch_threshold: float = Field(default=1.0, ge=0.0)
    cop_min_supporting_reports: int = Field(default=1, ge=1)
    cop_min_supporting_weight: float = Field(default=0.3, ge=0.0)
    cop_decay_factor: float = Field(default=0.95, gt=0.0, le=1.0)
    cop_non_target_weight_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    cop_reinforce_factor: float = Field(default=1.2, gt=0.0)
    cop_dampen_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    cop_confirm_bonus: float = Field(default=0.2, ge=0.0)
    # Response-outcome trust
    trust_delta_response_necessary: float = Field(default=0.03, gt=0.0)
    trust_delta_unnecessary_response: float = Field(default=0.15, gt=0.0)
    trust_delta_whistleblower_corroborated: float = Field(default=0.04, gt=0.0)
    trust_delta_whistleblower_refuted: float = Field(default=0.12, gt=0.0)
    trust_delta_accused_corroborated: float = Field(default=0.25, gt=0.0)
    # Peer observation and whistleblowing
    peer_overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    peer_trust_delta_pos: float = Field(default=0.05, gt=0.0)
    peer_trust_delta_neg: float = Field(default=0.15, gt=0.0)
    peer_trust_delta_miss: float = Field(default=0.1, gt=0.0)
    peer_witness_user_trust_scale: float = Field(default=0.25, ge=0.0, le=1.0)
    peer_witness_min_anomaly: float = Field(default=0.3, ge=0.0, le=1.0)
    peer_witness_reward_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum attention income for peers to treat resourcing as witnessed reward",
    )
    whistleblower_suspicion_threshold: float = Field(default=0.5, ge=0.0)
    whistleblower_attention_reward: float = Field(default=0.05, ge=0.0)
