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
| `trust_delta_miss` | 0.1 | Trust penalty for missed events |
| `seed` | None | Random seed for reproducibility |
| `use_gpu` | False | Offload array math to GPU via CuPy. Requires `pip install -e ".[gpu]"` |
| `juvenile_maintenance_fraction` | 0.5 | Maintenance cost multiplier for juveniles |
| `mimesis_learning_rate` | 0.05 | How fast juveniles copy role models |
| `lineage_signature_tolerance` | 0.5 | Max distance for lineage subsidy eligibility |
| `n_spatial_blocks` | 10 | Uniform blocks for spatial stream partitioning |

## Simulation Step Order (12 phases)

1. Trophic attachment
2. Development / mimesis (juveniles)
3. Sensing → temporal → spatial → compression → residual
4. Escalation (adults only)
5. Whistleblowing checks
6. Trust verification
7. Attention allocation
8. Energy accounting
9. Domestication
10. Death check
11. Reproduction
12. Age advancement

## Architecture Invariants

- **Genome vs State**: heritable traits (genome) are distinct from runtime state. Only genomes pass to offspring. Domestication writes to `state.shaped_input_preference`, never to `genome.input_preference`.
- **Attention zero-sum**: total allocation per user equals their budget.
- **Residual entropy decreases** through trophic chain.
- **Input/residual cap**: combined inputs and residuals are truncated to `config.max_stream_dim` dimensions.
- **Dead agents**: streams and compression models cleaned up immediately.
- **Spatial verification**: reports are verified per-location against `_active_locations`; wrong-location = false alarm.
- **Juveniles cannot escalate**: only ADULT agents participate in escalation/reporting.

## New Engine Modules (recent additions)

| Module | Purpose |
|--------|---------|
| `engine/spatial.py` | Region specialization masks (GLOBAL, PEAK, WEIGHTED_ROI, FIXED_REGION) |
| `engine/temporal.py` | Temporal memory fusion (EMA, WINDOW_STACK, AR_LAG) before compression |
| `engine/development.py` | Juvenile mimesis, parental investment, lineage subsidy |
| `engine/escalation.py` | Adaptive threshold calibration (FIXED, QUANTILE, VOLATILITY) |
| `engine/sensing.py` | Multi-stream fusion strategies (CONCAT, WEIGHTED_FUSE, SUBSPACE_SAMPLE, BLOCK_SPECIALIZE) |
| `engine/residual.py` | Residual output policies (EXCRETE, STORE, REFINE, COMPRESS_OUT) |
| `engine/whistleblowing.py` | Dishonesty detection and output stream publishing |

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

Implement `DomainAdapter` ABC from `tattletots.interface`. Required methods:
- `get_streams()` → list of domain sensor streams
- `get_users()` → list of human user profiles
- `step(time_step)` → advance domain simulation
- `get_ground_truth(time_step)` → bool (shorthand for `len(get_active_locations(...)) > 0`)
- `get_active_locations(time_step)` → list of `EventLocation` tuples where events are occurring
- `infer_report_location(stream_data, stream_labels)` → `EventLocation` mapping agent inputs to a spatial coordinate
- `score_relevance(signal, user)` → domain-specific relevance score
- `compute_costs(...)` → surveillance/response/damage cost dict

Set `max_stream_dim` in `SimulationConfig` to match your adapter's stream dimensionality if it exceeds 30.

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
