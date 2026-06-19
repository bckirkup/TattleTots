# Cross-Repository Coordination Guide

TattleTots is designed to work as a standalone engine or integrated with domain-specific simulation repositories. This document explains how to coordinate the four repos.

## Repository Ecosystem

| Repository | Role | CLI Command |
|------------|------|-------------|
| **TattleTots** | Domain-agnostic agent ecology engine | `tattletots` |
| **Coral_Key_in_Three_Hour_Epochs** | ReefWatch fishery monitoring domain | `coral-key` |
| **Xylella_SPQR** | GrainGuard precision agriculture domain | `grain-guard` |
| **Scrapiron_and_the_Bear** | FireEcology wildfire management domain | `fire-ecology` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TattleTots Engine                         │
│  (World, Agents, Streams, Trust, Compression, Evolution)     │
├─────────────────────────────────────────────────────────────┤
│                    DomainAdapter ABC                          │
├──────────────┬──────────────────┬───────────────────────────┤
│  Coral Key   │  Xylella_SPQR    │  Scrapiron_and_the_Bear   │
│  (ReefWatch) │  (GrainGuard)    │  (FireEcology)            │
└──────────────┴──────────────────┴───────────────────────────┘
```

Each domain repo:
- Implements the `DomainAdapter` abstract base class from TattleTots
- Can run independently (domain-only mode) via its own CLI
- Can be plugged into TattleTots for full agent ecology via `scripts/run_with_tattletots.py`

## Installation

### All repos together (recommended for cross-domain analysis)

```bash
# Clone all repos
git clone https://github.com/bckirkup/TattleTots.git
git clone https://github.com/bckirkup/Coral_Key_in_Three_Hour_Epochs.git
git clone https://github.com/bckirkup/Xylella_SPQR.git
git clone https://github.com/bckirkup/Scrapiron_and_the_Bear.git

# Install TattleTots first (all domain repos depend on it)
pip install -e TattleTots[dev]

# Install domain repos (they reference TattleTots via git dependency,
# but a local editable install takes precedence)
pip install -e Coral_Key_in_Three_Hour_Epochs[dev]
pip install -e Xylella_SPQR[dev]
pip install -e Scrapiron_and_the_Bear[dev]
```

### Single domain repo

```bash
pip install -e TattleTots[dev]
pip install -e <domain_repo>[dev]
```

## Running Modes

### 1. TattleTots Standalone (built-in Gaussian shift scenario)

```bash
tattletots --config configs/balanced.json --output results.json --verbose
```

Tests the engine with an abstract Gaussian distribution shift — no domain knowledge needed.

### 2. Domain Standalone (no agent ecology)

Each domain has its own CLI for exercising domain logic in isolation:

```bash
# Fishery monitoring
coral-key --epochs 200 --verbose

# Precision agriculture
grain-guard --steps 200 --landscape monoculture --verbose

# Wildfire management
fire-ecology --steps 200 --verbose
```

### 3. Integrated Mode (domain + TattleTots agent ecology)

The full integration runs the domain adapter inside the TattleTots World engine.
Agents evolve, compete, and self-organize to monitor the domain.

```bash
# From the domain repo directory:
python scripts/run_with_tattletots.py \
    --config configs/tattletots_integration.json \
    --output results.json \
    --verbose
```

## Configuration

Integrated configs combine two sections:

```json
{
  "simulation": {
    "initial_population": 20,
    "max_population": 60,
    "seed": 42,
    "max_stream_dim": 30,
    "..."
  },
  "domain": {
    "...domain-specific parameters..."
  }
}
```

- `simulation` → TattleTots engine parameters (see `tattletots.engine.config.SimulationConfig`)
- `domain` → Domain-specific parameters (varies per repo)

## Unified Output Schema

All integrated runs produce JSON conforming to `tattletots.output_schema.SimulationOutput`:

```json
{
  "schema_version": "1.0",
  "timestamp": "2025-01-01T00:00:00+00:00",
  "run_summary": {
    "domain": "fire_ecology",
    "steps_completed": 200,
    "seed": 42,
    "wall_time_seconds": 3.4
  },
  "simulation_config": { "..." },
  "domain_config": { "..." },
  "ecology_metrics": {
    "final_population": 18,
    "peak_population": 25,
    "total_births": 42,
    "total_deaths": 44,
    "total_reports": 156,
    "precision": 0.73,
    "max_trophic_depth": 2.5,
    "reached_equilibrium": true
  },
  "cost_metrics": {
    "total_surveillance_cost": 1200.0,
    "total_response_cost": 800.0,
    "total_damage_cost": 3500.0,
    "total_cost": 5500.0,
    "mean_cost_per_step": 27.5
  },
  "domain_metrics": { "...domain-specific..." },
  "time_series": {
    "population": [20, 19, 21, "..."],
    "cost_per_step": [12.5, 15.0, "..."],
    "reports_issued": [3, 1, "..."],
    "correct_reports": [2, 1, "..."],
    "false_alarms": [1, 0, "..."],
    "missed_events": [0, 1, "..."],
    "mean_info_energy": [1.2, 1.1, "..."],
    "mean_attn_energy": [0.8, 0.9, "..."],
    "births": [1, 0, "..."],
    "deaths": [0, 1, "..."],
    "n_compression_types": [4, 4, "..."],
    "max_trophic_level": [2.0, 2.5, "..."]
  }
}
```

### Cross-Domain Comparison

The `ecology_metrics` and `cost_metrics` sections are consistent across all domains,
enabling direct comparison:

```python
import json
from pathlib import Path
from tattletots.output_schema import SimulationOutput

results = []
for path in Path("outputs/").glob("*.json"):
    results.append(SimulationOutput.read_json(path))

# Compare precision across domains
for r in results:
    print(f"{r.run_summary.domain}: precision={r.ecology_metrics.precision:.2%}")
```

## Key Design Decisions

1. **TattleTots is never modified by domain repos** — they only implement the ABC
2. **Domain repos remain independently runnable** — the standalone CLI still works
3. **Unified output enables apples-to-apples comparison** — same structure regardless of domain
4. **Configuration is self-contained** — one JSON file per run with all parameters
5. **Domain metrics are extensible** — the `domain_metrics` field is free-form per domain
