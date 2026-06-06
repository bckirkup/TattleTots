"""Attention allocation: users distribute cognitive bandwidth to agents."""

from __future__ import annotations

import numpy as np

from tattletots.models.agent import Agent
from tattletots.models.user import User


def allocate_attention(user: User, agents: list[Agent]) -> dict[str, float]:
    """Softmax attention allocation from a user across agents.

    α_{k,i}(t) = A_k(t) * (τ_{k,i} * r_{k,i}) / Σ_j(τ_{k,j} * r_{k,j})

    Attention is zero-sum: total allocated ≤ user's budget.
    """
    if not agents:
        return {}

    scores: list[float] = []
    for agent in agents:
        trust = user.get_trust(agent.id)
        relevance = user.compute_relevance(agent.state.signal_vector)
        scores.append(trust * max(relevance, 0.0))

    total_score = sum(scores)
    if total_score <= 0:
        # Equal split if no differentiation
        equal_share = user.attention_budget / len(agents)
        return {agent.id: equal_share for agent in agents}

    allocations: dict[str, float] = {}
    for agent, score in zip(agents, scores, strict=True):
        allocations[agent.id] = user.attention_budget * (score / total_score)

    return allocations


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


def compute_niche_overlap(agent_a: Agent, agent_b: Agent) -> float:
    """Cosine similarity between two agents' signal vectors.

    High overlap → direct competition for the same attention niche.
    """
    sig_a = agent_a.state.signal_vector
    sig_b = agent_b.state.signal_vector

    if sig_a.size == 0 or sig_b.size == 0:
        return 0.0

    min_dim = min(len(sig_a), len(sig_b))
    a = sig_a[:min_dim]
    b = sig_b[:min_dim]

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
