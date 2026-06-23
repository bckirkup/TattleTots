---
name: tattletots-domain-adapter-guide
description: Shared reference for implementing and testing TattleTots DomainAdapter integrations. Use when creating a new domain adapter, debugging integration issues, or running cross-repo integrated simulations.
---

# TattleTots Domain Adapter Guide

## Overview

Domain repos implement the `DomainAdapter` ABC from `tattletots.interface.domain_adapter`
and plug into the TattleTots engine via `domain-runner`. This skill documents the shared
patterns across all domain adapters.

## Install Order (all domain repos follow this)

```bash
pip install -e domain-runner[dev]   # shared layer orchestration
pip install -e TattleTots[dev]      # engine (only for --layer tattletots)
pip install -e <domain_repo>[dev]   # the specific domain
pre-commit install
```

## DomainAdapter ABC — Required Methods

All domain adapters must implement these methods from `tattletots.interface.domain_adapter.DomainAdapter`:

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_streams()` | `list[Stream]` | Domain sensor streams (capped at `max_stream_dim` dims each) |
| `get_users()` | `list[User]` | Human user profiles with role priorities |
| `step(time_step)` | `None` | Advance domain simulation by one step |
| `get_ground_truth(time_step)` | `bool` | Whether events are currently active |
| `get_active_locations(time_step)` | `list[EventLocation]` | Spatial coordinates of active events |
| `infer_report_location(stream_data, stream_labels)` | `EventLocation` | Map agent inputs to spatial coordinate |
| `score_relevance(signal, user)` | `float` | Role-weighted relevance for COP fusion |
| `compute_costs(...)` | `dict` | Surveillance + response + damage cost breakdown |
| `get_responder_user_id()` | `str` | User authorized to receive COP dispatch targets |
| `dispatch_and_judge_responses(targets, time_step)` | `list[ResponseOutcome]` | Execute physical responses, return outcomes |

## Integration Loop Pattern

The integration loop uses:
```python
world.set_event_state(adapter.get_active_locations(step))
```

**NOT** `set_ground_truth`. The engine verifies report correctness per-location.

## Critical Invariants

- **Agents must NOT read `User.trust`** — trust is user-side only; agents use peer_trust + observable signals
- **Streams capped at `config.max_stream_dim` dimensions** (default 30, tunable)
- **`score_relevance()`** uses band-aligned role relevance via `tattletots.engine.relevance`
- **Output conforms to** `tattletots.output_schema.SimulationOutput` (unified JSON)

## Running Integrated Mode

```bash
# Each domain repo provides a CLI with --layer flag:
<domain-cli> sim --layer domain_only --steps N --verbose    # domain physics only
<domain-cli> sim --layer tattletots --config configs/tattletots_integration.json  # full ecology
<domain-cli> batch --config configs/batch_example.json      # parameter sweeps
```

## GPU Acceleration

```bash
pip install -e ".[gpu]"  # installs cupy-cuda12x
```

Set `"use_gpu": true` in the `"simulation"` section of the integration config.
Falls back silently to NumPy if CuPy or CUDA is unavailable.

Key files in TattleTots:
- `src/tattletots/engine/gpu_utils.py` — `get_array_module()`, `to_numpy()`, `gpu_available()`
- Compression, attention, and niche overlap use `xp = get_array_module(use_gpu)` pattern

## Parameter Scans

```bash
python scripts/run_with_tattletots.py --config <variant>.json --output results/<name>.json
```

Load results:
```python
from tattletots.output_schema import SimulationOutput
result = SimulationOutput.model_validate_json(path.read_text())
```

## Baselines

Each domain repo has a `baselines/` directory containing:
- `run_<domain>_baselines.py` — Parameter scan runner for A0-A3 architectures
- `<domain>_baselines_config.json` — Scan configuration
- `<domain>_baselines_results.zip` — Pre-computed reference results

## COP Relevance and Dispatch

Agent reports carry **compressed** `signal_vector`s. Domain users define **raw-stream** role
priority bands. At setup, `align_user_priorities_to_report_space()` resamples priorities to
the median agent working dimension.

During fusion, `fuse_reports_into_cops()` calls `adapter.score_relevance(signal, user)`
(not a raw prefix dot product).

Default: `tattletots.engine.relevance.band_relevance()` — maps each compressed component
to the proportional priority band. Domain adapters may override for custom role logic.

Dispatch gates on responder COP `threat_level` at **reported locations** (not ground truth).

## Adding a New Sensor (common pattern)

1. Create `src/<domain>/sensors/<sensor_name>.py` with Pydantic model
2. Implement `observe()` → `np.ndarray` (or `None` when off-cadence)
3. Wire into adapter: add to `_setup_streams()` dimensionality and `_update_streams()` data flow
4. Add tests in `tests/test_sensors/`
5. Update `sensors/__init__.py`

## Adding a New Architecture (common pattern)

1. Subclass `Architecture` from `architectures/base.py`
2. Implement `step()` returning architecture-specific result and `reset()`
3. Give identical sensor/hardware access as other architectures (no strawmen)
4. Add tests in `tests/test_architectures.py`
5. Update `architectures/__init__.py`

## Domain Repos

| Repo | Domain | CLI | Key Streams |
|------|--------|-----|-------------|
| `Scrapiron_and_the_Bear` | Wildfire | `fire-ecology` | thermal, weather, fuel moisture |
| `Xylella_SPQR` | Agriculture | `grain-guard` | pest, satellite, pheromone, soil |
| `Coral_Key_in_Three_Hour_Epochs` | Fishery/IUU | `coral-key` | AIS, SAR, catch, ocean, eDNA, EM |

## Coordination

See `docs/COORDINATION.md` in each domain repo and `docs/domain_integration.md` in TattleTots.
