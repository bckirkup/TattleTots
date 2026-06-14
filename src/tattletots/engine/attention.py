"""Attention allocation: users distribute cognitive bandwidth to agents."""

from __future__ import annotations

import numpy as np

from tattletots.engine.gpu_utils import get_array_module
from tattletots.models.agent import Agent
from tattletots.models.user import User


def allocate_attention(
    user: User, agents: list[Agent], use_gpu: bool = False
) -> dict[str, float]:
    """Softmax attention allocation from a user across agents.

    α_{k,i}(t) = A_k(t) * (τ_{k,i} * r_{k,i}) / Σ_j(τ_{k,j} * r_{k,j})

    Attention is zero-sum: total allocated ≤ user's budget.
    """
    if not agents:
        return {}

    xp = get_array_module(use_gpu)
    trust_arr = xp.array([user.get_trust(a.id) for a in agents], dtype=xp.float64)
    relevance_arr = xp.array(
        [max(user.compute_relevance(a.state.signal_vector), 0.0) for a in agents],
        dtype=xp.float64,
    )
    scores = trust_arr * relevance_arr
    total_score = float(xp.sum(scores))

    if total_score <= 0:
        equal_share = user.attention_budget / len(agents)
        return {agent.id: equal_share for agent in agents}

    weights = scores / total_score
    return {
        agent.id: user.attention_budget * float(weights[i])
        for i, agent in enumerate(agents)
    }


def compute_attention_income(
    agent: Agent,
    users: list[User],
    allocations: dict[str, dict[str, float]],
    verified_value: float = 1.0,
) -> float:
    """Total attention income for an agent from all users.

    income = Σ_k α_{k,i} * v_{k,i}
    """
    income = 0.0
    for user in users:
        user_alloc = allocations.get(user.id, {})
        alpha = user_alloc.get(agent.id, 0.0)
        income += alpha * verified_value
    return income


def compute_niche_overlap(
    agent_a: Agent, agent_b: Agent, use_gpu: bool = False
) -> float:
    """Cosine similarity between two agents' signal vectors.

    High overlap → direct competition for the same attention niche.
    """
    sig_a = agent_a.state.signal_vector
    sig_b = agent_b.state.signal_vector

    if sig_a.size == 0 or sig_b.size == 0:
        return 0.0

    xp = get_array_module(use_gpu)
    min_dim = min(len(sig_a), len(sig_b))
    a = xp.asarray(sig_a[:min_dim], dtype=xp.float64)
    b = xp.asarray(sig_b[:min_dim], dtype=xp.float64)

    norm_a = float(xp.linalg.norm(a))
    norm_b = float(xp.linalg.norm(b))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(xp.dot(a, b) / (norm_a * norm_b))
