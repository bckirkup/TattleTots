"""Agent genome: heritable traits that define an agent's behavior and structure."""

from __future__ import annotations

import enum
from typing import Self

import numpy as np
from pydantic import BaseModel, Field


class CompressionType(enum.StrEnum):
    """Supported compression model classes."""

    PCA = "pca"
    AR1 = "ar1"
    THRESHOLD = "threshold"
    WAVELET = "wavelet"


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

    def mutate(self, rng: np.random.Generator, rate: float = 0.1) -> Self:
        """Return a mutated copy of this genome."""
        data = self.model_dump()

        if rng.random() < rate:
            types = list(CompressionType)
            data["compression_type"] = types[int(rng.integers(0, len(types)))]

        if rng.random() < rate:
            data["n_components"] = int(np.clip(data["n_components"] + rng.integers(-2, 3), 1, 50))

        if rng.random() < rate:
            data["escalation_threshold"] = float(
                np.clip(data["escalation_threshold"] + rng.normal(0, 0.05), 0.0, 1.0)
            )

        if rng.random() < rate:
            data["metabolic_efficiency"] = float(
                np.clip(data["metabolic_efficiency"] + rng.normal(0, 0.1), 0.1, 5.0)
            )

        if rng.random() < rate:
            data["compute_cost"] = float(
                np.clip(data["compute_cost"] + rng.normal(0, 0.02), 0.01, 1.0)
            )

        if rng.random() < rate:
            data["maintenance_cost"] = float(
                np.clip(data["maintenance_cost"] + rng.normal(0, 0.01), 0.01, 0.5)
            )

        if rng.random() < rate:
            data["reproduction_threshold"] = float(
                np.clip(data["reproduction_threshold"] + rng.normal(0, 0.2), 0.5, 10.0)
            )

        if rng.random() < rate:
            data["domestication_sensitivity"] = float(
                np.clip(data["domestication_sensitivity"] + rng.normal(0, 0.05), 0.0, 1.0)
            )

        if rng.random() < rate:
            strategies = list(ParentalStrategy)
            data["parental_strategy"] = strategies[int(rng.integers(0, len(strategies)))]

        if rng.random() < rate:
            data["parental_investment"] = float(
                np.clip(data["parental_investment"] + rng.normal(0, 0.05), 0.0, 1.0)
            )

        if rng.random() < rate:
            modes = list(MimesisMode)
            data["mimesis_mode"] = modes[int(rng.integers(0, len(modes)))]

        if rng.random() < rate:
            data["lineage_signature"] = float(data["lineage_signature"] + rng.normal(0, 0.1))

        # Mutate input preferences if they exist
        if len(data["input_preference"]) > 0:
            pref = np.array(data["input_preference"], dtype=np.float64)
            mask = rng.random(len(pref)) < rate
            pref[mask] += rng.normal(0, 0.1, size=int(mask.sum()))
            pref = np.clip(pref, 0.0, None)
            total = pref.sum()
            if total > 0:
                pref /= total
            data["input_preference"] = pref

        # Mutate user affinity if it exists
        if len(data["target_user_affinity"]) > 0:
            aff = np.array(data["target_user_affinity"], dtype=np.float64)
            mask = rng.random(len(aff)) < rate
            aff[mask] += rng.normal(0, 0.1, size=int(mask.sum()))
            aff = np.clip(aff, 0.0, None)
            total = aff.sum()
            if total > 0:
                aff /= total
            data["target_user_affinity"] = aff

        return type(self).model_validate(data)

    @classmethod
    def recombine(cls, parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Self:
        """Sexual recombination: crossover of two parent genomes."""
        data_a = parent_a.model_dump()
        data_b = parent_b.model_dump()
        child_data: dict[str, object] = {}

        for key in data_a:
            if key in ("input_preference", "target_user_affinity"):
                arr_a = np.array(data_a[key], dtype=np.float64)
                arr_b = np.array(data_b[key], dtype=np.float64)
                if len(arr_a) == len(arr_b) and len(arr_a) > 0:
                    alpha = rng.random()
                    child_data[key] = alpha * arr_a + (1 - alpha) * arr_b
                else:
                    child_data[key] = arr_a if rng.random() < 0.5 else arr_b
            else:
                child_data[key] = data_a[key] if rng.random() < 0.5 else data_b[key]

        return cls.model_validate(child_data)
