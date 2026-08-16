# Cross-Repository Coordination Guide

TattleTots is a domain-agnostic engine. Coral Key, GrainGuard, and FireEcology
implement the `DomainAdapter` interface in their own repositories and use
`domain-runner` to select a domain-only or integrated layer.

## Repository ecosystem

| Repository | Role | CLI |
| --- | --- | --- |
| `domain-runner` | Shared single/batch runner | library |
| `TattleTots` | Agent ecology engine | `tattletots` |
| `Coral_Key_in_Three_Hour_Epochs` | ReefWatch fishery adapter | `coral-key` |
| `Xylella_SPQR` | GrainGuard agriculture adapter | `grain-guard` |
| `Scrapiron_and_the_Bear` | FireEcology wildfire adapter | `fire-ecology` |

Each repository is independently runnable. Its lockfile supplies the
`domain-runner` and `tattletots` dependencies; do not install sibling
repositories into one shared editable environment.

## Setup

Run the setup command from the repository named in each heading.

### TattleTots repository

From `TattleTots/`:

```bash
uv sync --locked --no-build --no-binary-package domain-runner --no-binary-package tattletots --extra dev
uv run --no-sync --no-build pre-commit install
```

### Domain repositories

From the relevant domain repository, use its own lockfile and package-specific
command:

```bash
# Coral_Key_in_Three_Hour_Epochs/
uv sync --locked --no-build --no-binary-package coral-key --no-binary-package domain-runner --no-binary-package tattletots --extra dev

# Xylella_SPQR/
uv sync --locked --no-build --no-binary-package grain-guard --no-binary-package domain-runner --no-binary-package tattletots --extra dev

# Scrapiron_and_the_Bear/
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev
```

The commands below are grouped by working directory. A command shown under a
domain repository is not intended to run from `TattleTots/`.

## Running TattleTots standalone

From `TattleTots/`, run one of the built-in scenarios:

```bash
uv run --no-sync --no-build tattletots --scenario gaussian_shift --steps 400 --verbose
uv run --no-sync --no-build tattletots --config configs/gaussian_shift_default.json --verbose
```

The CLI supports `gaussian_shift`, `high_dim_shift`, and `sparse_sensor`.

## Running a domain repository

From the relevant domain repository:

```bash
# Domain-only physics
uv run --no-sync --no-build fire-ecology sim --layer domain_only --steps 200 --verbose
uv run --no-sync --no-build grain-guard sim --layer domain_only --steps 200 --verbose
uv run --no-sync --no-build coral-key sim --layer domain_only --epochs 200 --verbose

# Integrated domain + TattleTots ecology
uv run --no-sync --no-build fire-ecology sim --layer tattletots --config configs/tattletots_integration.json
uv run --no-sync --no-build grain-guard sim --layer tattletots --config configs/tattletots_integration.json
uv run --no-sync --no-build coral-key sim --layer tattletots --config configs/tattletots_integration.json

# Batch sweeps
uv run --no-sync --no-build fire-ecology batch --config configs/batch_example.json
uv run --no-sync --no-build grain-guard batch --config configs/batch_example.json
uv run --no-sync --no-build coral-key batch --config configs/batch_example.json
```

Each domain repository also contains its own
`scripts/run_with_tattletots.py` compatibility wrapper. From that domain
repository, the equivalent invocation is:

```bash
uv run --no-sync --no-build python scripts/run_with_tattletots.py \
  --config configs/tattletots_integration.json \
  --output results.json \
  --verbose
```

The wrapper is not a TattleTots file and cannot be run from `TattleTots/`.

## Architecture and adapter contract

The engine calls the domain adapter through
`tattletots.interface.domain_adapter.DomainAdapter`. Adapters provide:

- `get_streams()` and `get_users()`;
- `step(time_step)`;
- `get_ground_truth(time_step)` and `get_active_locations(time_step)`;
- `infer_report_location(stream_data, stream_labels)`;
- `score_relevance(signal, user)`;
- `compute_costs(...)`;
- `get_responder_user_id()` and `dispatch_and_judge_responses(...)`.

Integrated execution records event locations with
`world.set_event_state(adapter.get_active_locations(step))`. Reports are
verified at their reported locations. Agents must not read `User.trust`; that
is user-side state used by attention and COP fusion.

The integrated flow is:

```text
TattleTotsLayer.setup()
  -> align_user_priorities_to_report_space()
world.step()
  -> run_dispatch_cycle()
       -> fuse_reports_into_cops(..., adapter=adapter)
       -> select_dispatch_targets()
       -> adapter.dispatch_and_judge_responses()
  -> apply_post_dispatch_feedback()
```

`fuse_reports_into_cops()` calls the adapter's
`score_relevance(signal, user)`. The default helpers are
`tattletots.engine.relevance.band_relevance` and
`score_report_relevance`.

## Integration configuration

An integrated configuration combines engine and domain sections:

```json
{
  "simulation": {
    "initial_population": 20,
    "max_population": 60,
    "max_stream_dim": 30,
    "mutation_rate": 0.1,
    "seed": 42
  },
  "domain": {
    "...domain-specific parameters..."
  }
}
```

The `max_population: 60` value above is a domain-integration override for
manageable cross-repository runs. The engine default is `100`, defined by
`SimulationConfig.max_population` in
`src/tattletots/engine/config.py`.

## Unified output

Integrated runs produce `tattletots.output_schema.SimulationOutput`. Read an
output file from the TattleTots repository with:

```bash
uv run --no-sync --no-build python -c \
  'from tattletots.output_schema import SimulationOutput; print(SimulationOutput.read_json("results.json"))'
```

The schema includes `run_summary`, `simulation_config`, `domain_config`,
`ecology_metrics`, `cost_metrics`, `domain_metrics`, and time-series data.
`ecology_metrics` and `cost_metrics` are shared across domain adapters, while
`domain_metrics` remains domain-specific.

## Cross-domain comparison

From `TattleTots/`, after placing output files there:

```bash
uv run --no-sync --no-build python -c '
from pathlib import Path
from tattletots.output_schema import SimulationOutput
for path in Path("outputs").glob("*.json"):
    result = SimulationOutput.read_json(path)
    print(f"{result.run_summary.domain}: precision={result.ecology_metrics.precision:.2%}")
'
```

The domain repositories share no implementation code with each other. They
share the adapter contract, `domain-runner`, and the output schema.

See [domain_integration.md](domain_integration.md) for implementation details.
