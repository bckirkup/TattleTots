# Architecture

## System Overview

TattleTots is a discrete-time simulation of an information ecology operating under dual-currency pressure. Each time step executes in a fixed order:

```
1. Trophic attachment  — Agents select input streams
2. Compression         — Agents extract structure, produce residuals
3. Escalation          — Adults decide whether to report anomalies
4. Trust verification  — Reports checked against ground truth
5. Attention allocation— Users distribute cognitive bandwidth
6. Energy accounting   — Dual-currency bookkeeping
7. Domestication       — Downstream→upstream shaping signals
8. Death check         — Starvation (either currency) → death
9. Reproduction        — Energy above threshold → offspring
10. Age advancement    — Juveniles mature, everyone ages
```

## Module Map

### models/ — Domain Types

| Module | Type | Responsibility |
|--------|------|---------------|
| `genome.py` | `Genome` | Heritable blueprint (immutable per agent, mutated at reproduction) |
| `agent.py` | `Agent`, `AgentState` | Living entity with genome + mutable state |
| `stream.py` | `Stream` | Multivariate time series (raw, residual, or output) |
| `user.py` | `User` | Human stakeholder with budget, priorities, trust |
| `energy.py` | `EnergyReserves` | Dual-currency accounting |
| `report.py` | `Report` | Escalation event directed at a user |

### engine/ — Simulation Core

| Module | Responsibility |
|--------|---------------|
| `world.py` | World state + step() orchestration |
| `config.py` | `SimulationConfig` parameters |
| `compression.py` | Pluggable compression models (PCA, AR1, Threshold) |
| `trophic.py` | Stream selection + trophic level computation |
| `attention.py` | Softmax allocation + niche overlap |
| `trust.py` | Asymmetric trust update on verification |
| `reproduction.py` | Asexual/sexual reproduction + population cap |
| `whistleblowing.py` | Dishonesty detection mechanics |
| `domestication.py` | Downstream→upstream shaping signals |

### interface/ — Domain Adapter

The `DomainAdapter` ABC defines the contract between TattleTots and domain simulations (FireEcology, CruiseEcology, etc.).

### scenarios/ — Built-in Tests

`GaussianShiftScenario` — K=10 structured Gaussian with midpoint regime change. Self-contained smoke test requiring no external domain.

### telemetry/ — Recording

`TelemetryRecorder` accumulates `StepRecord` per time step. Provides analytics: population history, stability detection, extinction cascade detection.

## Key Design Decisions

1. **No global optimizer.** Each agent acts locally. Global performance is emergent.
2. **Trophic levels measured, not assigned.** Computed from input graph topology.
3. **Trust emerges from dynamics, not from a penalty parameter.** The alarm asymmetry is a consequence of trust economics.
4. **Domain-agnostic core.** Streams are abstract multivariate time series. The engine doesn't know what the numbers mean.
5. **Genome vs State separation.** Heritable traits (genome) are distinct from runtime state (energy, connections, model weights). Only genomes pass to offspring.
