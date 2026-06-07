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
        ...

    def score_relevance(self, signal_vector, user) -> float:
        """How relevant is this compressed signal to this user?"""
        ...

    def compute_costs(self, n_escalations, n_correct, n_false_alarms, n_missed):
        """Return domain-specific cost breakdown."""
        ...
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

### 4. Provide Ground Truth

The engine uses ground truth to verify escalation reports. A `True` return
means a real event is occurring — agents that escalate correctly earn trust;
agents that stay silent lose trust.

```python
def get_ground_truth(self, time_step: int) -> bool:
    return self._events[time_step]  # Your domain-specific event schedule
```

### 5. Score Relevance

Domain-specific relevance scoring. How useful is a compressed signal to a
particular user?

```python
def score_relevance(self, signal_vector, user) -> float:
    # Example: cosine similarity between signal and user's priority
    cos_sim = np.dot(signal_vector, user.priority_vector) / (
        np.linalg.norm(signal_vector) * np.linalg.norm(user.priority_vector) + 1e-10
    )
    return max(float(cos_sim), 0.0)
```

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

for step_num in range(config.max_steps):
    adapter.step(step_num)
    world.set_ground_truth(adapter.get_ground_truth(step_num))
    world.step()
    if world.living_population == 0:
        break

# Analyze results
print(world.telemetry.summary())
```

## Using Cost Accounting

To track domain costs alongside the simulation:

```python
from tattletots.telemetry import CostAccumulator

costs = CostAccumulator()

for step_num in range(config.max_steps):
    adapter.step(step_num)
    world.set_ground_truth(adapter.get_ground_truth(step_num))
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
