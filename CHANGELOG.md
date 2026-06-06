# Changelog

All notable changes to TattleTots are documented here.

## [Unreleased]

### Added
- **Phase 7 — Mathematical invariant tests** (`tests/test_math_properties.py`)
  - Chain depth bounded by signal rank (§6.2)
  - Branching topologies more resilient than linear (§6.3)
  - H-D-W equilibrium — diverse escalation strategies coexist (§6.4)
  - Domestication effective only with signal overlap (§6.5)
  - Genome immutability verified under domestication
- **Phase 8 — Cost accounting** (`telemetry/cost_accounting.py`)
  - `StepCosts` dataclass for per-step cost breakdown
  - `CostAccumulator` with history tracking and summary analytics
- **Phase 8 — Extended telemetry**
  - Energy flow tracking: info yield, attention income, compute/maintenance cost
  - Demographic tracking: juveniles, adults, mean generation, compression type diversity
  - `energy_flow_history()` and `demographic_history()` on `TelemetryRecorder`
- **Phase 9 — Documentation**
  - `docs/theory.md`: Dual-currency model theoretical foundation
  - `docs/domain_integration.md`: How to write a domain adapter
  - Updated `docs/architecture.md` with Phase 8 modules

### Fixed
- **Double metabolic efficiency** — `_apply_energy` in `world.py` was applying
  `metabolic_efficiency` twice (once in compression, once in energy accounting),
  resulting in efficiency² scaling
- **Missing dimension cap in `_maybe_escalate`** — multi-stream agents could
  produce input vectors exceeding the 30-dim cap, causing shape mismatches in
  compression models
- **Domestication mutated genome** — `apply_shaping` was writing to
  `genome.input_preference` instead of `state.input_preference_override`,
  violating the genome-vs-state separation and causing Lamarckian inheritance
- **Sexual reproduction cost non-deterministic** — cost was computed from
  child's post-mutation threshold instead of parents' average threshold
- **PCA yielded zero on single-sample input** — PCA now maintains a sliding
  window of 20 samples for SVD, producing valid yield from step 1
- **PCA anomaly_score dimension mismatch** — added guard for input
  dimensionality changes between fit and score calls
- **Wolf agent uninitialized in smoke test** — added `_init_agent_model(wolf)`
  so the false-alarm test actually exercises the escalation path
- **Residual entropy test fragile** — switched from mean to median for
  cross-level variance comparison to reduce outlier sensitivity
- **Pre-existing mypy error** — `_random_genome` returned `str` instead of
  `CompressionType` for `compression_type` field

## [0.1.0] — 2026-06-04

### Added
- **Phase 1** — Project scaffolding, CI pipeline, pre-commit hooks
- **Phase 2** — Core domain models (Agent, Genome, Stream, User, Energy, Report)
- **Phase 3** — Simulation engine (World, compression, trophic, energy accounting, death)
- **Phase 4** — Attention allocation, trust dynamics, escalation mechanics
- **Phase 5** — Reproduction (asexual + sexual), mutation, development lifecycle
- **Phase 6** — Whistleblowing, domestication, competitive exclusion
- **Phase 7** — Smoke tests for 5 emergent behavior criteria + 2 math invariants
- **Scenario** — Gaussian shift scenario (K=10, midpoint regime change, 2 users)
- **CLI** — `tattletots` command with `--scenario` and `--config` options
- **Domain adapter ABC** — `DomainAdapter` interface for domain repos
- **Telemetry** — `StepRecord` + `TelemetryRecorder` with stability detection
