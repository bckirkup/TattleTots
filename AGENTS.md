# AGENTS.md — AI Agent Guidelines for TattleTots

## Repository Purpose
Simulation engine for dual-currency information ecologies. Domain-agnostic core
that will integrate with domain-specific repos (FireEcology, CruiseEcology, etc.).

## Setup
```bash
pip install -e domain-runner[dev]   # required for tests/test_dispatch_integration.py
pip install -e ".[dev]"
pre-commit install
```

## Validation Commands
Run these before committing:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest
```

## Architecture Rules
- **Never hardcode domain knowledge** (no fire/ship/network assumptions)
- **Never modify tests to make them pass** — fix the implementation
- **Genome fields are heritable** — runtime state is NOT
- **Compression models must handle variable-dimension input gracefully** (reset on dim change)
- **Agents never read `User.trust`** — user-side attention/COP only; agents use peer_trust + observable signals

## Key Files
| File | Purpose |
|------|---------|
| `src/tattletots/engine/world.py` | Main simulation loop (12-phase step) |
| `src/tattletots/models/genome.py` | Heritable traits + mutation/recombination |
| `src/tattletots/models/location.py` | `EventLocation` type alias for spatial coordinates |
| `src/tattletots/engine/compression.py` | Pluggable compression models |
| `src/tattletots/engine/spatial.py` | Spatial region specialization (mask + infer location) |
| `src/tattletots/engine/temporal.py` | Temporal memory fusion (EMA, window stack, AR lag) |
| `src/tattletots/engine/development.py` | Juvenile mimesis + parental investment |
| `src/tattletots/engine/escalation.py` | Adaptive threshold calibration |
| `src/tattletots/engine/sensing.py` | Multi-stream fusion strategies |
| `src/tattletots/engine/residual.py` | Residual output policies |
| `src/tattletots/engine/whistleblowing.py` | Dishonesty detection |
| `src/tattletots/engine/cop.py` | User COP fusion; calls `adapter.score_relevance()` |
| `src/tattletots/engine/relevance.py` | Band relevance + priority remapping for compressed reports |
| `src/tattletots/engine/dispatch_integration.py` | COP-gated dispatch cycle |
| `src/tattletots/engine/trust.py` | Peer trust and whistleblower verification |
| `src/tattletots/integration/tattletots_layer.py` | domain-runner SimulationLayer |
| `src/tattletots/interface/domain_adapter.py` | DomainAdapter ABC (incl. dispatch hooks) |
| `src/tattletots/scenarios/gaussian_shift.py` | Built-in smoke test scenario |
| `tests/test_smoke.py` | Success criteria validation |

## Performance Notes
- Streams must be cleaned up when agents die (orphan cleanup in step loop)
- Trophic level computation is O(N) with memoization — called every step
- Population cap prevents runaway reproduction
- Dimensionality cap (`max_stream_dim` in `SimulationConfig`) on inputs/residuals prevents exponential vector growth; tunable per scenario

## PR Requirements
- All ruff checks pass
- mypy strict passes on src/
- All tests pass (including smoke tests)
- New features include tests
- Update README if adding new scenarios or domain adapters
