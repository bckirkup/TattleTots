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
├── models/          # Domain types: Agent, Stream, User, Genome, Energy
├── engine/          # Simulation core: World, compression, trophic, trust
├── interface/       # DomainAdapter ABC for plugging in domain repos
├── scenarios/       # Built-in test scenarios (Gaussian shift)
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
    "seed": 42
  },
  "scenario": {
    "n_components": 10,
    "dimensionality": 20,
    "shift_step": 200
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

class FireEcologyAdapter(DomainAdapter):
    def get_streams(self) -> list[Stream]: ...
    def get_users(self) -> list[User]: ...
    def step(self, time_step: int) -> None: ...
    def get_ground_truth(self, time_step: int) -> bool: ...
    def score_relevance(self, signal, user) -> float: ...
    def compute_costs(self, ...) -> dict[str, float]: ...
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
