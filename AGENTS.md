# AGENTS.md — AI Agent Guidelines for TattleTots

## Repository Purpose
Simulation engine for dual-currency information ecologies. Domain-agnostic core
that will integrate with domain-specific repos (FireEcology, CruiseEcology, etc.).

## Setup
```bash
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
- **Input/residual streams are capped at `config.max_stream_dim`** (default 30) to prevent exponential blowup

## Key Files
| File | Purpose |
|------|---------|
| `src/tattletots/engine/world.py` | Main simulation loop (10-phase step) |
| `src/tattletots/models/genome.py` | Heritable traits + mutation/recombination |
| `src/tattletots/engine/compression.py` | Pluggable compression models |
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
