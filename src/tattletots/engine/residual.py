"""Residual excretion, storage, and refinement policies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tattletots.engine.compression import CompressionModel
from tattletots.models.agent import Agent
from tattletots.models.genome import ResidualPolicy


def apply_residual_policy(
    agent: Agent,
    residual: NDArray[np.float64],
    info_yield: float,
    *,
    refine_model: CompressionModel | None = None,
    max_dim: int = 256,
) -> tuple[NDArray[np.float64], float, int]:
    """Process residual according to genome policy.

    Returns (output_vector, adjusted_yield, output_dim).
    """
    genome = agent.genome
    policy = genome.residual_policy

    if policy == ResidualPolicy.EXCRETE:
        out = residual[:max_dim]
        return out, info_yield, out.size

    if policy == ResidualPolicy.STORE:
        buffer = agent.state.residual_buffer
        buffer.append(residual.copy())
        max_store = genome.residual_storage_steps
        if len(buffer) > max_store:
            agent.state.residual_buffer = buffer[-max_store:]
            buffer = agent.state.residual_buffer
        if len(buffer) >= max(1, max_store):
            averaged = np.mean(np.stack(buffer, axis=0), axis=0)
            agent.state.residual_buffer.clear()
            out = averaged[:max_dim]
            return out, info_yield * 0.9, out.size
        # Not yet emitting — publish zeros to avoid misleading downstream
        dim = min(residual.size, max_dim)
        return np.zeros(dim, dtype=np.float64), info_yield * 0.5, dim

    if policy == ResidualPolicy.REFINE and refine_model is not None:
        refined, extra_yield = refine_model.fit_transform(residual)
        # Invariant: refinement cannot increase structured variance beyond input
        refined_var = float(np.var(refined))
        input_var = float(np.var(residual))
        if refined_var > input_var:
            refined = residual * (input_var / max(refined_var, 1e-10)) ** 0.5
        out = refined[:max_dim]
        return out, info_yield + extra_yield * 0.5, out.size

    if policy == ResidualPolicy.COMPRESS_OUT:
        n = min(genome.n_components, residual.size, max_dim)
        summary = residual[:n]
        return summary, info_yield * 0.8, n

    out = residual[:max_dim]
    return out, info_yield, out.size
