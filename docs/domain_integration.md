# Domain Integration Guide

Domain repositories integrate with TattleTots by implementing
`tattletots.interface.domain_adapter.DomainAdapter`. TattleTots remains
domain-agnostic; domain physics, sensors, and response costs stay in the
domain repository.

## Repository setup and command context

Run TattleTots commands from `TattleTots/`:

```bash
uv sync --locked --no-build --no-binary-package domain-runner --no-binary-package tattletots --extra dev
uv run --no-sync --no-build pre-commit install
```

Run domain commands from the relevant domain repository. Each domain's
lockfile installs its package together with `domain-runner` and `tattletots`:

```bash
# From Scrapiron_and_the_Bear/
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev
uv run --no-sync --no-build fire-ecology sim --layer domain_only --steps 200
uv run --no-sync --no-build fire-ecology sim --layer tattletots --config configs/tattletots_integration.json
uv run --no-sync --no-build fire-ecology batch --config configs/batch_example.json
```

The analogous commands from `Coral_Key_in_Three_Hour_Epochs/` use `coral-key`
and its lockfile; those from `Xylella_SPQR/` use `grain-guard` and its
lockfile. The compatibility wrapper
`scripts/run_with_tattletots.py` lives in each domain repository, not in
TattleTots:

```bash
# From a domain repository
uv run --no-sync --no-build python scripts/run_with_tattletots.py \
  --config configs/tattletots_integration.json \
  --output results.json
```

Do not use the old multi-repository editable-install ritual. A repository's
committed `uv.lock` is the source of its dependency versions.

## The DomainAdapter interface

```python
from tattletots.interface import DomainAdapter
from tattletots.models.stream import Stream
from tattletots.models.user import User

class MyDomainAdapter(DomainAdapter):
    def get_streams(self) -> list[Stream]: ...
    def get_users(self) -> list[User]: ...
    def step(self, time_step: int) -> None: ...
    def get_ground_truth(self, time_step: int) -> bool: ...
    def get_active_locations(self, time_step: int) -> list[tuple[int, int]]: ...
    def infer_report_location(self, stream_data, stream_labels) -> tuple[int, int]: ...
    def score_relevance(self, signal_vector, user) -> float: ...
    def compute_costs(self, ...) -> dict: ...
    def get_responder_user_id(self) -> str: ...
    def dispatch_and_judge_responses(self, targets, time_step): ...
```

Import `DispatchTarget` from `tattletots.models.dispatch_target` and
`ResponseOutcome` from `tattletots.models.response_outcome` when annotating
dispatch methods. Agents must not read `User.trust`; agents learn from
peer-trust and observable reports or response outcomes.

The integration loop records spatial event state with:

```python
world.set_event_state(adapter.get_active_locations(step))
```

It verifies reports against active locations, rather than using a global
boolean alone.

## domain-runner layers

| Layer | Requires TattleTots | Behavior |
| --- | --- | --- |
| `domain_only` | No | Advance domain physics |
| `tattletots` | Yes | Agent ecology, COP fusion, and dispatch |

Domain repositories expose `*DomainHooks` and a `run_*_simulation()` entry
point. For example, from `Scrapiron_and_the_Bear/`:

```python
from fire_ecology.runner import FireDomainHooks, run_fire_simulation

hooks = FireDomainHooks()
run = hooks.load_run_context(
    cli_overrides={"layer": "tattletots", "domain": {"steps": 200}}
)
result = run_fire_simulation(run)
```

## Adapter implementation

### Streams and users

Streams have fixed dimensionality and expose NumPy data. Use
`StreamType.RAW` for basal domain sources; residual and output streams are
created by the engine. Combined inputs are capped at
`SimulationConfig.max_stream_dim`, whose default is 30.

Users have finite attention budgets and role-specific priority vectors:

```python
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User

Stream(name="sensor_array", stream_type=StreamType.RAW, dimensionality=20)
User(name="field_commander", attention_budget=1.0,
     priority_vector=np.array([0.8, 0.2]))
```

### State and location

`step()` advances domain state and updates each stream. `get_ground_truth()`
is a convenience boolean; `get_active_locations()` supplies the coordinates
used for report verification. `infer_report_location()` maps agent input to
the reported coordinate.

### Relevance and COP fusion

Reports carry compressed `signal_vector`s, while user priorities originate in
raw-stream space. COP fusion calls `adapter.score_relevance()` rather than
using a raw prefix dot product:

```python
from tattletots.engine.relevance import score_report_relevance

def score_relevance(self, signal_vector, user) -> float:
    return score_report_relevance(signal_vector, user)
```

At integrated setup, `align_user_priorities_to_report_space()` maps priorities
to the median agent working dimension. Override the default for domain-specific
role logic.

### Costs and dispatch

Return surveillance, response, and damage categories from `compute_costs()`:

```python
def compute_costs(self, n_escalations, n_correct, n_false_alarms, n_missed):
    return {
        "surveillance_cost": n_escalations * self.cost_per_escalation,
        "response_cost": n_correct * self.cost_per_response,
        "damage_cost": n_missed * self.cost_per_miss,
    }
```

`get_responder_user_id()` identifies the user authorized to dispatch physical
responses. `dispatch_and_judge_responses()` executes selected targets and
returns `ResponseOutcome` values.

## Manual loop

For low-level experiments, run this code from the TattleTots repository after
the TattleTots setup above:

```python
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World

config = SimulationConfig(
    initial_population=20,
    max_population=100,
    max_steps=500,
    max_stream_dim=50,
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
```

The `max_population=100` value is the engine default. Domain integration
configs may override it (for example, `60`) to keep cross-repository runs
manageable.

## Reference implementation

`scenarios/gaussian_shift.py` is the built-in complete adapter example. It
generates structured Gaussian components, a midpoint regime shift, and two
distinct users. See [COORDINATION.md](COORDINATION.md) for repository
architecture, setup, output schema, and comparison guidance.
