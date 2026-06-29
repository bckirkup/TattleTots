"""Agent genome: heritable traits that define an agent's behavior and structure."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tattletots.engine.config import GenePoolConfig, SimulationConfig


class CompressionType(enum.StrEnum):
    """Supported compression model classes."""

    PCA = "pca"
    AR1 = "ar1"
    THRESHOLD = "threshold"
    WAVELET = "wavelet"


class SensingStrategy(enum.StrEnum):
    """How an agent fuses multiple input streams into a working vector."""

    CONCAT = "concat"
    WEIGHTED_FUSE = "weighted_fuse"
    SUBSPACE_SAMPLE = "subspace_sample"
    BLOCK_SPECIALIZE = "block_specialize"


class TemporalFusionMode(enum.StrEnum):
    """How temporal history is fused before compression."""

    NONE = "none"
    EMA = "ema"
    WINDOW_STACK = "window_stack"
    AR_LAG = "ar_lag"


class SpatialStrategy(enum.StrEnum):
    """How an agent specializes spatially over input dimensions."""

    GLOBAL = "global"
    PEAK = "peak"
    WEIGHTED_ROI = "weighted_roi"
    FIXED_REGION = "fixed_region"


class ResidualPolicy(enum.StrEnum):
    """What an agent does with compression residuals."""

    EXCRETE = "excrete"
    STORE = "store"
    REFINE = "refine"
    COMPRESS_OUT = "compress_out"


class EscalationMode(enum.StrEnum):
    """How escalation threshold is determined."""

    FIXED = "fixed"
    ADAPTIVE_QUANTILE = "adaptive_quantile"
    ADAPTIVE_VOLATILITY = "adaptive_volatility"


class ParentalStrategy(enum.StrEnum):
    """Reproductive investment strategies (evolvable).

    EGG: Minimal investment. Many cheap offspring, high mortality.
    LIVE_BIRTH: Heavy investment. Energy subsidy during juvenile phase.
    MARSUPIAL: Intermediate. Parent provides a curated stream for juvenile training.
    """

    EGG = "egg"
    LIVE_BIRTH = "live_birth"
    MARSUPIAL = "marsupial"


class MimesisMode(enum.StrEnum):
    """How a juvenile learns what to learn during development.

    NONE: No observation. Pure genome priors + random exploration.
    PARENTAL: Observe only declared parents' behavior (input choices, compression).
    NICHE: Observe any agent consuming similar streams (same trophic neighborhood).
    OPPORTUNISTIC: Observe any high-trust agent (copy the successful, regardless of lineage).
    """

    NONE = "none"
    PARENTAL = "parental"
    NICHE = "niche"
    OPPORTUNISTIC = "opportunistic"


_ARRAY_MUTATION_KEYS = (
    "input_preference",
    "target_user_affinity",
    "fusion_weights",
    "region_affinity",
)


def _mutate_enum_field(
    data: dict[str, Any],
    rng: np.random.Generator,
    rate: float,
    key: str,
    enum_cls: type[enum.StrEnum],
) -> None:
    if rng.random() < rate:
        values = list(enum_cls)
        data[key] = values[int(rng.integers(0, len(values)))]


def _mutate_scalar_fields(data: dict[str, Any], rng: np.random.Generator, rate: float) -> None:
    if rng.random() < rate:
        data["n_components"] = int(np.clip(data["n_components"] + rng.integers(-2, 3), 1, 50))
    if rng.random() < rate:
        data["escalation_threshold"] = float(
            np.clip(float(data["escalation_threshold"]) + rng.normal(0, 0.05), 0.0, 1.0)
        )
    if rng.random() < rate:
        data["metabolic_efficiency"] = float(
            np.clip(float(data["metabolic_efficiency"]) + rng.normal(0, 0.1), 0.1, 5.0)
        )
    if rng.random() < rate:
        data["compute_cost"] = float(
            np.clip(float(data["compute_cost"]) + rng.normal(0, 0.02), 0.01, 1.0)
        )
    if rng.random() < rate:
        data["maintenance_cost"] = float(
            np.clip(float(data["maintenance_cost"]) + rng.normal(0, 0.01), 0.01, 0.5)
        )
    if rng.random() < rate:
        data["reproduction_threshold"] = float(
            np.clip(float(data["reproduction_threshold"]) + rng.normal(0, 0.2), 0.5, 10.0)
        )
    if rng.random() < rate:
        data["domestication_sensitivity"] = float(
            np.clip(float(data["domestication_sensitivity"]) + rng.normal(0, 0.05), 0.0, 1.0)
        )
    if rng.random() < rate:
        data["parental_investment"] = float(
            np.clip(float(data["parental_investment"]) + rng.normal(0, 0.05), 0.0, 1.0)
        )
    if rng.random() < rate:
        data["lineage_signature"] = float(float(data["lineage_signature"]) + rng.normal(0, 0.1))
    if rng.random() < rate:
        data["working_dim"] = int(np.clip(int(data["working_dim"]) + rng.integers(-8, 9), 8, 256))
    if rng.random() < rate:
        data["dim_offset"] = int(np.clip(int(data["dim_offset"]) + rng.integers(-5, 6), 0, 1000))
    if rng.random() < rate:
        data["block_index"] = int(np.clip(int(data["block_index"]) + rng.integers(-2, 3), 0, 99))
    if rng.random() < rate:
        data["temporal_memory_depth"] = int(
            np.clip(int(data["temporal_memory_depth"]) + rng.integers(-5, 6), 0, 100)
        )
    if rng.random() < rate:
        row, col = data["spatial_region"]
        data["spatial_region"] = (
            int(np.clip(int(row) + rng.integers(-2, 3), 0, 99)),
            int(np.clip(int(col) + rng.integers(-2, 3), 0, 99)),
        )
    if rng.random() < rate:
        data["spatial_radius"] = int(
            np.clip(int(data["spatial_radius"]) + rng.integers(-1, 2), 0, 20)
        )
    if rng.random() < rate:
        data["residual_storage_steps"] = int(
            np.clip(int(data["residual_storage_steps"]) + rng.integers(-2, 3), 0, 20)
        )
    if rng.random() < rate:
        data["escalation_memory_depth"] = int(
            np.clip(int(data["escalation_memory_depth"]) + rng.integers(-10, 11), 3, 200)
        )
    if rng.random() < rate:
        data["threshold_adaptation_rate"] = float(
            np.clip(float(data["threshold_adaptation_rate"]) + rng.normal(0, 0.05), 0.0, 1.0)
        )


def _mutate_array_preferences(data: dict[str, Any], rng: np.random.Generator, rate: float) -> None:
    for key in _ARRAY_MUTATION_KEYS:
        arr = np.array(data[key], dtype=np.float64)
        if len(arr) == 0:
            continue
        mask = rng.random(len(arr)) < rate
        arr[mask] += rng.normal(0, 0.1, size=int(mask.sum()))
        arr = np.clip(arr, 0.0, None)
        total = arr.sum()
        if total > 0 and key != "region_affinity":
            arr /= total
        data[key] = arr


class Genome(BaseModel):
    """Heritable blueprint for an agent's behavior.

    The genome specifies model class, hyperparameters, input preferences,
    escalation threshold, and target user. Mutations and recombination
    operate on this structure.
    """

    model_config = {"arbitrary_types_allowed": True}

    compression_type: CompressionType = Field(
        default=CompressionType.PCA,
        description="Which compression model class this agent uses",
    )
    n_components: int = Field(default=3, ge=1, le=50, description="Number of components to extract")
    input_preference: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Weight vector over available streams (learned/evolved)",
    )
    escalation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Anomaly score threshold above which the agent escalates",
    )
    target_user_affinity: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Affinity vector toward each user (softmax over users for report routing)",
    )
    development_duration: int = Field(
        default=5, ge=1, description="Time steps from birth to adulthood"
    )
    metabolic_efficiency: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplier on compression yield (higher = better at extracting structure)",
    )
    compute_cost: float = Field(default=0.1, gt=0.0, description="Per-step information energy cost")
    maintenance_cost: float = Field(
        default=0.05, gt=0.0, description="Per-step attention energy cost"
    )
    reproduction_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum combined energy to reproduce",
    )
    domestication_sensitivity: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="How much this agent responds to downstream shaping signals",
    )
    parental_strategy: ParentalStrategy = Field(
        default=ParentalStrategy.EGG,
        description="Investment strategy for offspring (egg=cheap, live_birth=subsidized, marsupial=curated stream)",
    )
    parental_investment: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of energy invested per offspring (live_birth) or stream cost (marsupial)",
    )
    mimesis_mode: MimesisMode = Field(
        default=MimesisMode.PARENTAL,
        description="How juvenile observes and learns during development",
    )
    lineage_signature: float = Field(
        default=0.0,
        description="Heritable signature for parent-offspring recognition (subsidy validation)",
    )
    # --- Compute complexity traits ---
    sensing_strategy: SensingStrategy = Field(
        default=SensingStrategy.CONCAT,
        description="How multiple input streams are fused before compression",
    )
    working_dim: int = Field(
        default=30,
        ge=8,
        le=256,
        description="Target dimensionality after sensing projection",
    )
    fusion_weights: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Per-stream fusion weights (normalized)",
    )
    dim_offset: int = Field(
        default=0,
        ge=0,
        description="Seed offset for subspace sampling (lineage diversity)",
    )
    block_index: int = Field(
        default=0,
        ge=0,
        description="Which spatial block to specialize on (block_specialize)",
    )
    temporal_memory_depth: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Depth of temporal history buffer",
    )
    temporal_fusion_mode: TemporalFusionMode = Field(
        default=TemporalFusionMode.NONE,
        description="How temporal history is fused",
    )
    spatial_strategy: SpatialStrategy = Field(
        default=SpatialStrategy.GLOBAL,
        description="Spatial specialization strategy",
    )
    spatial_region: tuple[int, int] = Field(
        default=(0, 0),
        description="Center of fixed spatial region (row, col)",
    )
    spatial_radius: int = Field(
        default=1,
        ge=0,
        description="Radius for fixed spatial region",
    )
    region_affinity: np.ndarray = Field(
        default_factory=lambda: np.array([], dtype=np.float64),
        description="Soft weights over domain regions",
    )
    residual_policy: ResidualPolicy = Field(
        default=ResidualPolicy.EXCRETE,
        description="How residuals are handled after compression",
    )
    residual_storage_steps: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Max steps to store residuals before emission",
    )
    escalation_mode: EscalationMode = Field(
        default=EscalationMode.FIXED,
        description="How escalation threshold is calibrated",
    )
    escalation_memory_depth: int = Field(
        default=50,
        ge=3,
        le=200,
        description="History window for escalation baseline",
    )
    threshold_adaptation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Rate for adaptive escalation threshold modes",
    )

    def total_compute_cost(
        self,
        config: SimulationConfig,
        *,
        residual_buffer_len: int = 0,
        residual_buffer_dim: int = 0,
    ) -> float:
        """Total per-step information-energy cost for compute complexity traits."""
        cost = self.compute_cost
        cost += self.temporal_memory_depth * config.temporal_cost_rate
        cost += self.working_dim * config.projection_cost_rate
        if self.spatial_strategy != SpatialStrategy.GLOBAL:
            cost += config.spatial_cost_rate
        if self.residual_policy == ResidualPolicy.STORE and residual_buffer_len > 0:
            cost += config.storage_cost_rate * residual_buffer_dim * residual_buffer_len
        if self.residual_policy == ResidualPolicy.REFINE:
            cost += self.compute_cost * config.refine_cost_multiplier
        if self.escalation_mode != EscalationMode.FIXED:
            cost += self.escalation_memory_depth * config.escalation_cost_rate
        return cost

    def mutate(self, rng: np.random.Generator, rate: float = 0.1) -> Self:
        """Return a mutated copy of this genome."""
        data = self.model_dump()
        _mutate_enum_field(data, rng, rate, "compression_type", CompressionType)
        _mutate_scalar_fields(data, rng, rate)
        _mutate_enum_field(data, rng, rate, "parental_strategy", ParentalStrategy)
        _mutate_enum_field(data, rng, rate, "mimesis_mode", MimesisMode)
        _mutate_enum_field(data, rng, rate, "sensing_strategy", SensingStrategy)
        _mutate_enum_field(data, rng, rate, "temporal_fusion_mode", TemporalFusionMode)
        _mutate_enum_field(data, rng, rate, "spatial_strategy", SpatialStrategy)
        _mutate_enum_field(data, rng, rate, "residual_policy", ResidualPolicy)
        _mutate_enum_field(data, rng, rate, "escalation_mode", EscalationMode)
        _mutate_array_preferences(data, rng, rate)
        return type(self).model_validate(data)

    @classmethod
    def recombine(cls, parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Self:
        """Sexual recombination: crossover of two parent genomes."""
        data_a = parent_a.model_dump()
        data_b = parent_b.model_dump()
        child_data: dict[str, object] = {}

        array_keys = (
            "input_preference",
            "target_user_affinity",
            "fusion_weights",
            "region_affinity",
        )
        for key in data_a:
            if key in array_keys:
                arr_a = np.array(data_a[key], dtype=np.float64)
                arr_b = np.array(data_b[key], dtype=np.float64)
                if len(arr_a) == len(arr_b) and len(arr_a) > 0:
                    alpha = rng.random()
                    child_data[key] = alpha * arr_a + (1 - alpha) * arr_b
                else:
                    child_data[key] = arr_a if rng.random() < 0.5 else arr_b
            elif key == "spatial_region":
                child_data[key] = data_a[key] if rng.random() < 0.5 else data_b[key]
            else:
                child_data[key] = data_a[key] if rng.random() < 0.5 else data_b[key]

        return cls.model_validate(child_data)

    @classmethod
    def random_genome(
        cls,
        rng: np.random.Generator,
        *,
        n_streams: int = 1,
        n_users: int = 1,
        gene_pool: GenePoolConfig | None = None,
    ) -> Genome:
        """Generate a random genome, optionally constrained by gene pool config."""
        from tattletots.engine.config import GenePoolConfig as PoolConfig

        pool: GenePoolConfig = gene_pool if gene_pool is not None else PoolConfig()
        comp_types = pool.available_compression_types or list(CompressionType)
        comp_type = CompressionType(comp_types[int(rng.integers(0, len(comp_types)))])
        lo, hi = pool.n_components_range
        et_lo, et_hi = pool.escalation_threshold_range
        me_lo, me_hi = pool.metabolic_efficiency_range
        wd_lo, wd_hi = pool.working_dim_range
        dd_lo, dd_hi = pool.development_duration_range

        sensing_types = pool.available_sensing_strategies or list(SensingStrategy)
        spatial_types = pool.available_spatial_strategies or [SpatialStrategy.GLOBAL]
        residual_types = pool.available_residual_policies or [ResidualPolicy.EXCRETE]
        escalation_types = pool.available_escalation_modes or [EscalationMode.FIXED]

        return cls(
            compression_type=comp_type,
            n_components=int(rng.integers(lo, hi + 1)),
            input_preference=rng.dirichlet(np.ones(max(n_streams, 1))),
            escalation_threshold=float(rng.uniform(et_lo, et_hi)),
            target_user_affinity=rng.dirichlet(np.ones(max(n_users, 1))),
            metabolic_efficiency=float(rng.uniform(me_lo, me_hi)),
            compute_cost=float(rng.uniform(0.05, 0.2)),
            maintenance_cost=float(rng.uniform(0.02, 0.1)),
            reproduction_threshold=float(rng.uniform(1.5, 3.0)),
            domestication_sensitivity=float(rng.uniform(0.0, 0.3)),
            development_duration=int(rng.integers(dd_lo, dd_hi + 1)),
            sensing_strategy=SensingStrategy(
                sensing_types[int(rng.integers(0, len(sensing_types)))]
            ),
            working_dim=int(rng.integers(wd_lo, wd_hi + 1)),
            fusion_weights=rng.dirichlet(np.ones(max(n_streams, 1))),
            dim_offset=int(rng.integers(0, 100)),
            block_index=int(rng.integers(0, max(pool.n_blocks, 1))),
            temporal_memory_depth=int(rng.integers(0, pool.max_temporal_depth + 1)),
            temporal_fusion_mode=TemporalFusionMode(
                pool.available_temporal_modes[
                    int(rng.integers(0, len(pool.available_temporal_modes)))
                ]
            ),
            spatial_strategy=SpatialStrategy(
                spatial_types[int(rng.integers(0, len(spatial_types)))]
            ),
            spatial_region=(int(rng.integers(0, 10)), int(rng.integers(0, 10))),
            spatial_radius=int(rng.integers(0, 5)),
            residual_policy=ResidualPolicy(
                residual_types[int(rng.integers(0, len(residual_types)))]
            ),
            escalation_mode=EscalationMode(
                escalation_types[int(rng.integers(0, len(escalation_types)))]
            ),
        )
