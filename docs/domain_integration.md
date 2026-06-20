# Domain Integration Guide

How to connect a domain-specific simulation to the TattleTots engine.

## Overview

TattleTots is domain-agnostic. The engine operates on abstract data streams,
agents, and users — it does not know whether the underlying domain involves
wildfire monitoring, ship biome analysis, or network intrusion detection.

Domain-specific behavior is provided by implementing the `DomainAdapter`
abstract base class.

## The DomainAdapter Interface

```python
from tattletots.interface import DomainAdapter
from tattletots.models.stream import Stream
from tattletots.models.user import User

class MyDomainAdapter(DomainAdapter):

    def get_streams(self) -> list[Stream]:
        """Create and return the domain's data streams."""
        ...

    def get_users(self) -> list[User]:
        """Define the human stakeholders in this domain."""
        ...

    def step(self, time_step: int) -> None:
        """Advance the domain state by one step.
        Update stream.current_data for each stream."""
        ...

    def get_ground_truth(self, time_step: int) -> bool:
        """Is a real event (fire, intrusion, etc.) active right now?"""
        return len(self.get_active_locations(time_step)) > 0

    def get_active_locations(self, time_step: int) -> list[tuple[int, int]]:
        """Return grid cells or zones where a true event is active."""
        ...

    def infer_report_location(
        self,
        stream_data: list,
        stream_labels: list[str],
    ) -> tuple[int, int]:
        """Infer the location an agent reports from its input stream data."""
        ...

    def score_relevance(self, signal_vector, user) -> float:
        """How relevant is this compressed signal to this user?"""
        ...

    def compute_costs(self, n_escalations, n_correct, n_false_alarms, n_missed):
        """Return domain-specific cost breakdown."""
        ...

    def get_responder_user_id(self) -> str:
        """User id authorized to dispatch physical responses from their COP."""

    def dispatch_and_judge_responses(
        self,
        targets: list[DispatchTarget],
        time_step: int,
    ) -> list[ResponseOutcome]:
        """Execute COP-selected responses and return outcome judgments."""
        ...
```

Import `DispatchTarget` from `tattletots.models.dispatch_target` and
`ResponseOutcome` from `tattletots.models.response_outcome`.

**Trust boundary:** Agents must not read `User.trust`. That field is for user-side
attention and COP fusion only. Agents update `peer_trust` from observable signals
(reports, dispatch outcomes, whistleblower verification).

## domain-runner Layers

Domain repos ship `{package}/runner.py` with a `*DomainHooks` class. The shared
[domain-runner](https://github.com/bckirkup/domain-runner) package orchestrates runs:

| Layer | Requires TattleTots | Behavior |
|-------|---------------------|----------|
| `domain_only` | No | Advance adapter physics only |
| `tattletots` | Yes | Full agent ecology + COP dispatch loop |

```bash
pip install -e domain-runner[dev]
pip install -e TattleTots[dev]   # only for --layer tattletots

fire-ecology sim --layer domain_only --steps 200
fire-ecology sim --layer tattletots --config configs/tattletots_integration.json
fire-ecology batch --config configs/batch_example.json
```

Programmatic use:

```python
from fire_ecology.runner import FireDomainHooks, run_fire_simulation

hooks = FireDomainHooks()
run = hooks.load_run_context(cli_overrides={"layer": "tattletots", "domain": {"steps": 200}})
result = run_fire_simulation(run)
```

## Step-by-Step Implementation

### 1. Define Your Streams

Each stream is a multivariate time series with fixed dimensionality.

```python
from tattletots.models.stream import Stream, StreamType

def get_streams(self) -> list[Stream]:
    return [
        Stream(
            name="sensor_array_1",
            stream_type=StreamType.RAW,
            dimensionality=20,
        ),
        Stream(
            name="sensor_array_2",
            stream_type=StreamType.RAW,
            dimensionality=15,
        ),
    ]
```

**Rules:**
- Each stream has a fixed `dimensionality` that never changes.
- Stream data (`current_data`) must be a NumPy array of that exact size.
- `StreamType.RAW` for basal data sources. Residual and output streams are
  created automatically by the engine.
- The engine caps combined agent inputs at `config.max_stream_dim` dimensions
  (default 30). If your streams exceed this, set `max_stream_dim` in the
  `SimulationConfig` to match your adapter's dimensionality.

### 2. Define Your Users

Users represent human stakeholders with finite attention budgets.

```python
from tattletots.models.user import User

def get_users(self) -> list[User]:
    return [
        User(
            name="field_commander",
            attention_budget=1.0,
            priority_vector=np.array([0.8, 0.2]),
        ),
        User(
            name="operations_center",
            attention_budget=1.5,
            priority_vector=np.array([0.3, 0.7]),
        ),
    ]
```

**Rules:**
- `attention_budget` is the total attention the user can allocate per step.
- `priority_vector` defines the user's relative interest in different signal
  dimensions. Length should match the number of streams.

### 3. Implement `step()`

Update stream data each time step. This is where your domain simulation runs.

```python
def step(self, time_step: int) -> None:
    for stream in self._streams:
        stream.current_data = self._generate_data(stream, time_step)
```

The engine calls `step()` once per simulation step, before agents consume
the streams.

### 4. Provide Ground Truth and Active Locations

The engine verifies escalation reports against **active event locations**.
Each report includes a `location` tuple `(row, col)` or `(zone_x, zone_y)`.
A report is correct iff its location is in the active set for that step.
Wrong-location reports are false alarms; agents that stay silent or report
the wrong location during an active event receive missed-event penalties.

```python
def get_ground_truth(self, time_step: int) -> bool:
    return len(self.get_active_locations(time_step)) > 0

def get_active_locations(self, time_step: int) -> list[tuple[int, int]]:
    return self._active_event_cells  # domain-specific

def infer_report_location(self, stream_data, stream_labels) -> tuple[int, int]:
    # Map agent input streams to the reported location
    ...
```

### 5. Score Relevance

Role-weighted relevance for COP fusion and attention. Agent reports use **compressed**
`signal_vector`s; user `priority_vector`s are defined in **raw stream space** with
role-specific bands (e.g. first/middle/last third).

**COP fusion calls `adapter.score_relevance()`**, not a blind dot product on the
first N components. The default helper is `tattletots.engine.relevance.score_report_relevance()`,
which uses proportional band mapping when dimensions differ.

At integrated setup, `TattleTotsLayer` remaps user priorities to the median agent
report dimension via `align_user_priorities_to_report_space()`.

```python
def score_relevance(self, signal_vector, user) -> float:
    from tattletots.engine.relevance import score_report_relevance
    return score_report_relevance(signal_vector, user)
```

Override when your domain has custom role logic (e.g. fisheries enforcement vs stock signals).
See `tests/test_relevance.py` for band-mapping examples.

### 6. Compute Domain Costs

Return a dictionary with three cost categories:

```python
def compute_costs(self, n_escalations, n_correct, n_false_alarms, n_missed):
    return {
        "surveillance_cost": n_escalations * self.cost_per_escalation,
        "response_cost": n_correct * self.cost_per_response,
        "damage_cost": n_missed * self.cost_per_miss,
    }
```

These costs feed into the `CostAccumulator` telemetry module for analysis.

## Running a Simulation with Your Adapter

### Manual loop (low-level)

```python
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World

config = SimulationConfig(
    initial_population=20,
    max_population=100,
    max_steps=500,
    max_stream_dim=50,  # raise from default 30 if your streams are high-dimensional
    seed=42,
)

adapter = MyDomainAdapter(...)
world = World(config=config)

for stream in adapter.get_streams():
    world.add_stream(stream)
for user in adapter.get_users():
    world.add_user(user)
world.seed_population()
world.set_location_inference(adapter.infer_report_location)

for step_num in range(config.max_steps):
    adapter.step(step_num)
    world.set_event_state(adapter.get_active_locations(step_num))
    world.step()
    if world.living_population == 0:
        break

# Analyze results
print(world.telemetry.summary())
```

### Recommended: domain-runner + TattleTots layer

Use each domain's `run_*_simulation()` entry point (see `{domain}/runner.py`) with
`--layer tattletots` or `TattleTotsLayer` directly. The layer runs `world.step()`,
then `run_dispatch_cycle()` and `apply_post_dispatch_feedback()` automatically.

See `integration/tattletots_layer.py` and `docs/COORDINATION.md`.

## Using Cost Accounting

To track domain costs alongside the simulation:

```python
from tattletots.telemetry import CostAccumulator

costs = CostAccumulator()

for step_num in range(config.max_steps):
    adapter.step(step_num)
    world.set_event_state(adapter.get_active_locations(step_num))
    world.step()

    # Get last step's telemetry
    last = world.telemetry.history[-1]
    cost_dict = adapter.compute_costs(
        n_escalations=last.reports_issued,
        n_correct=last.correct_reports,
        n_false_alarms=last.false_alarms,
        n_missed=0,  # compute from ground truth vs reports
    )
    costs.record_from_dict(step_num, cost_dict)

print(costs.summary())
```

## Reference Implementation

See `scenarios/gaussian_shift.py` for a complete working adapter. It generates
K=10 structured Gaussian components with noise, a midpoint regime shift, and
two distinct users.
