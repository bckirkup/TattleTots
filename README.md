# TattleTots

**Domain-agnostic simulation engine for dual-currency information ecologies.**

TattleTots models populations of information-processing agents ("Tots") that compete in an evolutionary ecology with two survival currencies: *information* (can you compress data?) and *attention* (do humans care about your reports?).

## Quick Start

```bash
pip install -e ".[dev]"
tattletots --scenario gaussian_shift --steps 400 --verbose
```

## What It Does

Tots consume data streams, compress them into residuals, form trophic hierarchies, escalate anomalies to human users, and evolve. The engine is domain-agnostic — it knows about streams, agents, and users, not about fires, ships, or networks.

### Dual-Currency Survival

Every agent maintains two energy reserves:

- **Information energy**: earned by compressing data (extracting structure), received as subsidy from downstream consumers
- **Attention energy**: earned by having reports valued by human users

An agent dies if *either* reserve hits zero. This creates the fundamental tension: you can be brilliant at compression but starve if nobody reads your reports, or popular but parasitic if you never actually extract structure.

### Emergent Behavior (Not Hardcoded)

- **Trophic hierarchies** emerge from agents choosing inputs that maximize yield
- **Specialization** emerges from niche partitioning across multiple users
- **Precision** emerges from trust dynamics (false alarms destroy funding)
- **Whistleblowing** emerges when agents profitably detect dishonesty

## Architecture

```
src/tattletots/
├── models/          # Domain types: Agent, Stream, User, Genome, Energy, Location
├── engine/          # Simulation core: World, compression, sensing, spatial,
│                    #   temporal, development, escalation, trophic, trust,
│                    #   reproduction, residual, whistleblowing, domestication
├── interface/       # DomainAdapter ABC for plugging in domain repos
├── scenarios/       # Built-in tests (Gaussian, high-dimensional, sparse sensors)
├── telemetry/       # History recording and analytics
└── cli.py           # Command-line entry point
```

## Configuration

Simulations are parameterized via JSON:

```json
{
  "simulation": {
    "initial_population": 20,
    "max_population": 100,
    "mutation_rate": 0.1,
    "max_stream_dim": 30,
    "seed": 42
  },
  "scenario": {
    "n_components": 10,
    "dimensionality": 20,
    "shift_step": 100,
    "total_steps": 200
  },
  "gene_pool": {
    "available_compression_types": ["pca", "ar1", "threshold"]
  }
}
```

Run with config: `tattletots --config configs/gaussian_shift_default.json --verbose`

## Development Lifecycle (Ontogeny)

New agents boot through a juvenile phase:

1. **Juvenile** — Agent consumes streams and trains its compression model, but cannot escalate or reproduce. Reduced energy drain (no attention maintenance cost).
2. **Niche discovery** — Input preferences + exploration determine which streams the juvenile attaches to. Bad matches → starvation before maturity.
3. **Graduation** — After `development_duration` steps, if alive → ADULT. Compression model is warm from training.
4. **Adult** — Full participation: escalation, reproduction, competition.

The evolutionary pressure operates on the boot process itself: genomes encoding model types mismatched to available signal structure produce juveniles that die young.

## Domain Integration

Domain repositories implement `DomainAdapter`:

```python
from tattletots.interface import DomainAdapter
from tattletots.models.location import EventLocation

class FireEcologyAdapter(DomainAdapter):
    def get_streams(self) -> list[Stream]: ...
    def get_users(self) -> list[User]: ...
    def step(self, time_step: int) -> None: ...
    def get_ground_truth(self, time_step: int) -> bool: ...
    def get_active_locations(self, time_step: int) -> list[EventLocation]: ...
    def infer_report_location(
        self, stream_data, stream_labels
    ) -> EventLocation | None: ...
    def score_relevance(self, signal, user) -> float: ...
    def compute_costs(self, ...) -> dict[str, float]: ...
```

The engine uses `get_active_locations()` for spatial report verification — each
agent report includes a location, and correctness is evaluated per-location
(not just globally). The `infer_report_location()` callback maps agent input
streams to a spatial coordinate. A decoder may return `None` when no usable
declared geometry is present; the engine preserves its existing spatial
fallback instead of treating abstention as the origin `(0, 0)`.

### Adapter Conformance

Domain repositories can validate an adapter's public instrument contract with
`validate_adapter_conformance(adapter, steps=...)` from
`tattletots.interface`. The structured report checks that:

- reflected sensor-like objects are actually observed and that published raw
  streams do not change without any sensor observation call;
- declared sensor coordinates round-trip through `infer_report_location`;
- stream metadata and status declarations are dimensionally consistent; and
- declared geometry, ground-truth locations, and decoded reports stay inside
  the public location frame.

The sensor-call and raw-stream check is a heuristic observation, not proof of
intent. It inspects adapter-owned raw streams only; residual and output streams
generated by the engine are excluded. Sensor reflection uses duck typing on
callable `observe`, `scan`, and `detect` methods and recursively handles
objects in nested lists, tuples, sets, and dictionaries. The tracker patches
the sensor **class** method temporarily rather than the instance method, and
restores it in `finally`; callers should therefore not run it concurrently
in-process with another test using the same sensor class.

State-independence requires an optional
`state_independence_factory` returning two adapters with matching sensor
configuration and different hidden state. Static sensor-inventory declarations
(`sensor_coordinates`, `footprints`, `resolution`, and `modality`) must always
match: instrument placement and capability cannot depend on what an instrument
observes. `identity` is also treated as static when it identifies a fixed
instrument. For a mobile or cooperative stream, an identity paired with a
changing reported coordinate is part of the observation and is reported with
the dynamic fields. Dynamic `coordinates` and status vectors describe
observations, so differences are reported informationally by default. This
allows cooperative or mobile instruments to report a moving vessel or to go
missing when it goes dark; those changes are evidence, not declaration leaks.
Pass `strict_state_independence=True` to require dynamic coordinates and
statuses to match too, which is appropriate for fixed-coverage instruments.
Without the factory the report explicitly records the check as **not
exercised** rather than passing it. `assert_adapter_conformance(...)` accepts
the same strict flag and is a thin pytest-friendly wrapper that raises an
assertion containing all failed findings.

### Available Domain Adapters

| Repository | Domain | Package |
|------------|--------|---------|
| [Coral_Key_in_Three_Hour_Epochs](https://github.com/bckirkup/Coral_Key_in_Three_Hour_Epochs) | Fishery monitoring & IUU detection | `coral-key` |
| [Xylella_SPQR](https://github.com/bckirkup/Xylella_SPQR) | Precision agriculture & pest management | `grain-guard` |
| [Scrapiron_and_the_Bear](https://github.com/bckirkup/Scrapiron_and_the_Bear) | Wildfire detection & suppression | `fire-ecology` |

Each domain repo includes `{package}/runner.py` and CLI commands (`sim`, `batch`, `--layer domain_only|tattletots`) via [domain-runner](https://github.com/bckirkup/domain-runner). Legacy `scripts/run_with_tattletots.py` wrappers still work. See [docs/COORDINATION.md](docs/COORDINATION.md) for installation, configuration, and output schema.

## GPU Acceleration

TattleTots supports optional GPU offloading via [CuPy](https://cupy.dev/) for large-population runs and parameter scans:

```bash
# Install with GPU support
pip install -e ".[gpu]"

# Enable in config JSON
{"simulation": {"use_gpu": true, ...}}

# Or pass directly via the engine
tattletots --config configs/gpu_scan.json --verbose
```

When `use_gpu: true`, all array math (compression SVD, attention softmax, niche overlap) dispatches to CuPy. Falls back to NumPy silently if CuPy is unavailable or no CUDA device is found.

### Parameter Scans

For large sweeps across parameter space, use the runner script with shell parallelism:

```bash
# Single run with JSON output
python scripts/run_with_tattletots.py --config configs/tattletots_integration.json --output results.json

# Parallel parameter scan (example: vary mutation_rate and seed)
for rate in 0.01 0.05 0.1 0.2 0.5; do
  for seed in $(seq 1 10); do
    python -c "
import json, sys
cfg = json.load(open('configs/gaussian_shift_default.json'))
cfg['simulation']['mutation_rate'] = $rate
cfg['simulation']['seed'] = $seed
json.dump(cfg, open(f'configs/scan/mr{rate}_s{seed}.json', 'w'))
" && python scripts/run_with_tattletots.py \
      --config "configs/scan/mr${rate}_s${seed}.json" \
      --output "results/mr${rate}_s${seed}.json" &
  done
done
wait
```

All output files conform to `tattletots.output_schema.SimulationOutput`, so results can be loaded and compared programmatically:

```python
import json, pathlib
from tattletots.output_schema import SimulationOutput

results = [
    SimulationOutput.model_validate_json(p.read_text())
    for p in pathlib.Path("results").glob("*.json")
]
```

## Testing

```bash
pytest                        # All tests
pytest -m smoke               # Smoke tests (emergent behavior validation)
pytest --cov=tattletots       # With coverage
```

### Success Criteria (Requirements §9)

The engine passes if:
1. Trophic hierarchies depth > 2 emerge from random seed
2. Population reaches stable equilibrium
3. Removing basal streams → upstream extinction cascades
4. False-alarm agents die; accurate agents reproduce
5. At least 2 distinct genome clusters coexist

## License

Apache 2.0
