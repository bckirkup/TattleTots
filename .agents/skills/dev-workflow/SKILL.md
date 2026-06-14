---
name: tattletots-dev-workflow
description: Development workflow, testing, and configuration reference for the TattleTots simulation engine.
---

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Validation (run before every commit)

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest
```

## Running a Simulation

```bash
tattletots --scenario gaussian_shift --steps 400 --verbose
tattletots --config configs/gaussian_shift_default.json --verbose
```

## Test Markers

```bash
pytest                   # all tests
pytest -m smoke          # emergent behavior validation only
pytest --cov=tattletots  # with coverage
```

## Key Configuration Parameters (`SimulationConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_population` | 100 | Hard population cap |
| `initial_population` | 20 | Starting agent count |
| `max_stream_dim` | 30 | Max dimensionality for agent inputs and residual streams. Increase for high-dimensional domain adapters. |
| `mutation_rate` | 0.1 | Genome mutation probability at reproduction |
| `recombination_probability` | 0.3 | Sexual vs asexual reproduction |
| `false_alarm_penalty` | 0.3 | Attention energy cost for false alarms |
| `trust_delta_pos` / `trust_delta_neg` | 0.05 / 0.2 | Asymmetric trust update magnitudes |
| `seed` | None | Random seed for reproducibility |
| `use_gpu` | False | Offload array math to GPU via CuPy. Requires `pip install -e ".[gpu]"` |

## Simulation Step Order (10 phases)

1. Trophic attachment
2. Compression
3. Escalation
4. Trust verification
5. Attention allocation
6. Energy accounting
7. Domestication
8. Death check
9. Reproduction
10. Age advancement

## Architecture Invariants

- **Genome vs State**: heritable traits (genome) are distinct from runtime state. Only genomes pass to offspring. Domestication writes to `state.shaped_input_preference`, never to `genome.input_preference`.
- **Attention zero-sum**: total allocation per user equals their budget.
- **Residual entropy decreases** through trophic chain.
- **Input/residual cap**: combined inputs and residuals are truncated to `config.max_stream_dim` dimensions.
- **Dead agents**: streams and compression models cleaned up immediately.

## GPU Acceleration

```bash
pip install -e ".[gpu]"  # installs cupy-cuda12x
```

Set `"use_gpu": true` in config JSON. All array math (SVD in PCA compression, attention
softmax, niche overlap cosine) dispatches to CuPy. Falls back silently to NumPy if no GPU.

Key files:
- `src/tattletots/engine/gpu_utils.py` — `get_array_module()`, `to_numpy()`, `gpu_available()`
- Compression, attention, and niche overlap use `xp = get_array_module(use_gpu)` pattern

## Adding Domain Adapters

Implement `DomainAdapter` ABC from `tattletots.interface`. Set `max_stream_dim` in `SimulationConfig` to match your adapter's stream dimensionality if it exceeds 30.

See `docs/domain_integration.md` for the full guide and `scenarios/gaussian_shift.py` for a reference implementation.

### Available Domain Repos

| Repo | Domain | Runner |
|------|--------|--------|
| `Coral_Key_in_Three_Hour_Epochs` | Fishery/IUU | `scripts/run_with_tattletots.py` |
| `Xylella_SPQR` | Precision agriculture | `scripts/run_with_tattletots.py` |
| `Scrapiron_and_the_Bear` | Wildfire | `scripts/run_with_tattletots.py` |

All runners output unified JSON (`tattletots.output_schema.SimulationOutput`).
See `docs/COORDINATION.md` for the full cross-repo guide.

## Parameter Scans

For large sweeps, generate config variants and run in parallel:

```bash
# Each run produces a SimulationOutput JSON file
python scripts/run_with_tattletots.py --config <cfg>.json --output <out>.json

# Load results programmatically
from tattletots.output_schema import SimulationOutput
result = SimulationOutput.model_validate_json(path.read_text())
```

Key parameters to sweep: `mutation_rate`, `initial_population`, `max_population`,
`false_alarm_penalty`, `trust_delta_neg`, `seed` (replicate).

## Pre-existing Configs

| Config | Use Case |
|--------|----------|
| `gaussian_shift_default.json` | Reference config with all params explicit |
| `balanced.json` | General-purpose, moderate false alarm rate |
| `conservative.json` | Low false alarm rate, harsh penalties |
| `harsh_ecology.json` | Strong selection pressure, high mutation |
| `longrun_evolution.json` | 1000-step runs for evolutionary dynamics |
| `sensitive.json` | High sensitivity, low penalties |
