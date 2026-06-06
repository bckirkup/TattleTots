# Dual-Currency Information Ecology — Theoretical Foundation

## Overview

TattleTots models information ecologies where autonomous agents compete for
survival under two independent currencies: **information energy** and
**attention energy**. Neither currency alone suffices — an agent must maintain
positive reserves of both to remain alive.

This document describes the mathematical framework that governs the simulation.

## The Two Currencies

### Information Energy

Information energy measures an agent's ability to extract structure from data.
An agent earns information energy by compressing its input stream(s) — the
higher the yield of its compression model, the more information energy it
receives per step.

```
info_delta = compression_yield(model, data) * metabolic_efficiency
```

**Sources:** compression yield, upstream subsidy from downstream consumers.
**Sinks:** compute cost, maintenance cost, reproduction cost.

### Attention Energy

Attention energy measures how much human stakeholders value the agent's reports.
Users allocate finite cognitive bandwidth across competing agents; an agent's
share depends on trust, signal relevance, and niche overlap with rivals.

```
attn_delta = user_allocation(trust, relevance, competition)
```

**Sources:** user attention allocation after verified escalations.
**Sinks:** maintenance cost, false-alarm penalties, reproduction cost.

### Dual-Currency Survival

An agent dies if either currency reaches zero. This creates a fundamental
tension:

- **Pure compression specialists** may extract structure brilliantly but starve
  for attention if they never escalate (or escalate poorly).
- **Attention parasites** may earn user funding temporarily but lose information
  energy if they lack genuine compression ability.

The viable strategy space is the intersection of both survival constraints.

## Trophic Structure

### Measured, Not Assigned

Trophic levels emerge from the input graph topology. They are not assigned by
the engine. An agent that consumes raw data streams has trophic level 1; an
agent consuming another agent's residual has trophic level 2; and so on.

```
trophic_level(agent) = 1 + mean(trophic_level(sources))
```

where `sources` are the agents producing the streams this agent consumes. Raw
streams have level 0.

### Trophic Chains and Signal Rank

The maximum depth of a trophic chain is bounded by the signal rank of the
basal data. With K structured components and compression models extracting
k components each, the theoretical chain depth is approximately `ceil(K/k)`.
Beyond this depth, no extractable structure remains in the residuals.

## Trust Dynamics

### Asymmetric Updates

Trust between a user and an agent updates asymmetrically:

- **Correct alarm:** `trust += Δ⁺` (small, e.g., 0.05)
- **False alarm:** `trust -= Δ⁻` (large, e.g., 0.20)
- **Missed event:** `trust -= Δ_miss` (moderate, e.g., 0.10)

Trust is hard to build and easy to destroy. This is not a parameter choice but
a reflection of real-world information economics: one false fire alarm costs
more credibility than ten correct ones earn.

### Precision as Emergent Property

Because `Δ⁻ >> Δ⁺`, agents face evolutionary pressure toward precision. Agents
with low escalation thresholds (many false alarms) lose trust, lose attention
energy, and die. This produces precision without any explicit precision
objective in the agent's genome.

## Genome vs State

### Heritable Traits (Genome)

The genome encodes fixed parameters that do not change during an agent's
lifetime and are passed (with possible mutation) to offspring:

- Compression model type and hyperparameters
- Input preferences (initial stream selection weights)
- Escalation threshold
- Target user affinity
- Metabolic efficiency
- Cost parameters
- Reproduction threshold

### Runtime State

State includes everything that changes during the agent's lifetime:

- Current energy reserves
- Active input stream connections
- Compression model weights (learned from data)
- Signal vector (current output)
- Domestication preference overrides
- Generation counter

**Critical invariant:** Only genomes pass to offspring. Runtime state is never
inherited. Domestication effects (niche construction) modify the parent's state,
not its genome, ensuring Darwinian rather than Lamarckian inheritance.

## Domestication (Niche Construction)

Downstream agents can shape upstream agents' compression behavior by sending
shaping signals. This is analogous to biological niche construction: a consumer
that finds certain features useful nudges its food source to produce more of
those features.

### Requirements for Effective Domestication

1. **Signal overlap:** Shaping only works when the signal dimensions of the
   downstream agent match the input preference dimensions of the upstream agent.
2. **Sensitivity > 0:** The upstream agent's `domestication_sensitivity` genome
   parameter must be positive.
3. **State modification only:** Domestication writes to
   `AgentState.input_preference_override`, never to the genome.

## Mathematical Invariants

The following properties are tested automatically (see `tests/test_math_properties.py`):

1. **Residual entropy decreases** through trophic chains (§6.1)
2. **Chain depth bounded** by signal rank `ceil(K/k)` (§6.2)
3. **Branching > linear stability** — redundant input sources provide
   resilience (§6.3)
4. **H-D-W equilibrium** — honest, deceiver, and whistleblower strategies
   coexist at equilibrium rather than one dominating (§6.4)
5. **Domestication requires overlap** — shaping only improves yield when signal
   components overlap (§6.5)
6. **Attention zero-sum** per user — total allocation equals budget (§6.6)

## Compression Models

The engine supports pluggable compression models:

| Model | Mechanism | Strengths |
|-------|-----------|-----------|
| PCA | SVD on sliding window, top-k components | Best for linearly structured signals |
| AR(1) | First-order autoregressive prediction | Best for temporally correlated data |
| Threshold | Simple anomaly detection via deviation | Robust to high-dimensional noise |
| Wavelet | Multi-scale decomposition (future) | Best for mixed-frequency signals |

Each model implements `fit_transform(data) -> (residual, yield)` and
`anomaly_score(data) -> float`. The PCA model maintains a sliding window
of recent samples to handle the engine's single-sample-per-step operation.

## Reproduction

Agents reproduce when their energy exceeds a genome-defined threshold. The
engine supports both asexual (clonal with mutation) and sexual (two-parent
crossover + mutation) reproduction.

**Cost determination:** For sexual reproduction, the energy cost per parent is
computed from the parents' average reproduction threshold — not the child's
post-mutation threshold. This prevents non-deterministic costs from child
mutations.
