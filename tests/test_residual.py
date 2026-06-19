"""Tests for residual policies."""

from __future__ import annotations

import numpy as np

from tattletots.engine.compression import PCACompression
from tattletots.engine.residual import apply_residual_policy
from tattletots.models.agent import Agent
from tattletots.models.genome import Genome, ResidualPolicy


class TestResidual:
    def test_excrete_passthrough(self) -> None:
        agent = Agent(genome=Genome(residual_policy=ResidualPolicy.EXCRETE))
        residual = np.arange(10.0)
        out, yield_, dim = apply_residual_policy(agent, residual, 0.5, max_dim=8)
        assert dim == 8
        assert out.size == 8
        assert yield_ == 0.5

    def test_store_buffers_then_emits(self) -> None:
        agent = Agent(genome=Genome(residual_policy=ResidualPolicy.STORE, residual_storage_steps=2))
        r = np.ones(5)
        apply_residual_policy(agent, r, 0.5, max_dim=5)
        out, _, dim = apply_residual_policy(agent, r * 2, 0.5, max_dim=5)
        assert dim == 5
        assert out.size == 5

    def test_refine_preserves_variance_bound(self) -> None:
        agent = Agent(genome=Genome(residual_policy=ResidualPolicy.REFINE, n_components=2))
        model = PCACompression(n_components=2)
        residual = np.random.default_rng(0).standard_normal(20)
        input_var = float(np.var(residual))
        out, _, _ = apply_residual_policy(agent, residual, 0.3, refine_model=model, max_dim=20)
        assert float(np.var(out)) <= input_var + 1e-6

    def test_compress_out_reduces_dim(self) -> None:
        agent = Agent(genome=Genome(residual_policy=ResidualPolicy.COMPRESS_OUT, n_components=3))
        residual = np.arange(20.0)
        out, _, dim = apply_residual_policy(agent, residual, 0.5, max_dim=20)
        assert dim == 3
        assert out.size == 3
