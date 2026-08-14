"""End-to-end guards for seeded simulation determinism."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tattletots.cli import _load_scenario
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.models.identity import stable_id_digest
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_payload(
    seed: int,
    steps: int = 8,
    *,
    reproduction_coupling_strength: float = 1.0,
    reproduction_information_scale: float = 1.0,
    reproduction_attention_scale: float = 1.0,
    grounding_quality_strength: float = 0.5,
) -> dict[str, object]:
    scenario = GaussianShiftScenario(seed=seed, total_steps=steps)
    config = SimulationConfig(
        initial_population=4,
        max_population=20,
        max_steps=steps,
        seed=seed,
        reproduction_coupling_strength=reproduction_coupling_strength,
        reproduction_information_scale=reproduction_information_scale,
        reproduction_attention_scale=reproduction_attention_scale,
        grounding_quality_strength=grounding_quality_strength,
    )
    world = World(config=config)
    for stream in scenario.get_streams():
        world.add_stream(stream)
    for user in scenario.get_users():
        world.add_user(user)
    world.seed_population()
    world.set_location_inference(scenario.infer_report_location)
    world.set_dim_to_location(scenario.dim_index_to_location)

    for step in range(steps):
        scenario.step(step)
        world.set_event_state(scenario.get_active_locations(step))
        world.step()

    return {
        "agents": sorted(world.agents),
        "streams": sorted(world.streams),
        "users": sorted(world.users),
        "records": [asdict(record) for record in world.telemetry.history],
    }


def _records(payload: dict[str, object]) -> list[dict[str, object]]:
    records = payload["records"]
    assert isinstance(records, list)
    return [record for record in records if isinstance(record, dict)]


def _mean_metric(payload: dict[str, object], key: str) -> float:
    values = [record[key] for record in _records(payload)]
    return sum(float(value) for value in values) / len(values)


def _sum_metric(payload: dict[str, object], key: str) -> int:
    return sum(int(record[key]) for record in _records(payload))


# Change-detector only, not a correctness check. Re-derive it from these IDs and
# integer telemetry fields if a structural change intentionally updates it.
def _structural_fingerprint(payload: dict[str, object]) -> str:
    """Hash stable IDs and integer telemetry as a change-detector only."""
    structural = {
        "agents": payload["agents"],
        "streams": payload["streams"],
        "users": payload["users"],
        "records": [
            {
                key: record[key]
                for key in (
                    "time_step",
                    "population",
                    "births",
                    "deaths",
                    "reports_issued",
                    "correct_reports",
                    "false_alarms",
                    "active_location_count",
                    "n_streams",
                )
                if key in record
            }
            for record in _records(payload)
        ],
    }
    encoded = json.dumps(structural, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# Invariants: seeded runs must reproduce stable structure and integer outcomes.
def test_same_seed_reproduces_the_complete_run() -> None:
    first = _structural_fingerprint(_run_payload(42))
    second = _structural_fingerprint(_run_payload(42))

    assert first == second


def test_same_seed_has_zero_fingerprint_spread_across_repeated_runs() -> None:
    fingerprints = {_structural_fingerprint(_run_payload(42)) for _ in range(10)}

    assert len(fingerprints) == 1


def test_different_seed_changes_the_complete_run() -> None:
    assert _structural_fingerprint(_run_payload(42)) != _structural_fingerprint(_run_payload(43))


def test_stable_id_digest_has_a_process_independent_golden_value() -> None:
    assert stable_id_digest("00000000-0000-4000-8000-000000000001") == 1289600646178507792


def test_cli_seed_reaches_the_builtin_scenario() -> None:
    actual = _load_scenario("gaussian_shift", {}, seed=7)
    expected = GaussianShiftScenario(seed=7)
    actual.step(0)
    expected.step(0)
    for actual_stream, expected_stream in zip(
        actual.get_streams(), expected.get_streams(), strict=True
    ):
        np.testing.assert_array_equal(actual_stream.current_data, expected_stream.current_data)


def test_same_seed_is_independent_of_python_hash_seed() -> None:
    script = """
import hashlib
import json
from dataclasses import asdict

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario


scenario = GaussianShiftScenario(seed=42, total_steps=8)
world = World(config=SimulationConfig(initial_population=4, max_population=20, max_steps=8, seed=42))
for stream in scenario.get_streams():
    world.add_stream(stream)
for user in scenario.get_users():
    world.add_user(user)
world.seed_population()
world.set_location_inference(scenario.infer_report_location)
world.set_dim_to_location(scenario.dim_index_to_location)
for step in range(8):
    scenario.step(step)
    world.set_event_state(scenario.get_active_locations(step))
    world.step()
payload = {
    "agents": sorted(world.agents),
    "streams": sorted(world.streams),
    "users": sorted(world.users),
    "records": [asdict(record) for record in world.telemetry.history],
}
structural = {
    "agents": payload["agents"],
    "streams": payload["streams"],
    "users": payload["users"],
    "records": [
        {
            key: record[key]
            for key in (
                "time_step",
                "population",
                "births",
                "deaths",
                "reports_issued",
                "correct_reports",
                "false_alarms",
                "active_location_count",
                "n_streams",
            )
        }
        for record in payload["records"]
    ],
}
print(hashlib.sha256(json.dumps(structural, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
"""
    fingerprints: list[str] = []
    for hash_seed in ("0", "1"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(  # noqa: S603 - fixed argv invokes the local test interpreter.
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        fingerprints.append(result.stdout.strip())

    assert fingerprints[0] == fingerprints[1]


# Sensitivity: mechanism-driven telemetry must move in the expected direction.
def test_reproduction_coupling_strength_reduces_births() -> None:
    values = (0.0, 0.5, 1.0)
    births = [
        _sum_metric(
            _run_payload(42, steps=40, reproduction_coupling_strength=value),
            "births",
        )
        for value in values
    ]

    assert births[0] > births[1] > births[2]
    assert births[0] - births[-1] >= 10


def test_grounding_quality_strength_increases_effective_grounded_share() -> None:
    values = (0.0, 0.25, 0.5)
    shares = [
        _mean_metric(
            _run_payload(42, steps=40, grounding_quality_strength=value),
            "effective_grounded_yield_share",
        )
        for value in values
    ]

    assert shares[0] < shares[1] < shares[2]
    assert shares[-1] - shares[0] >= 0.1


def test_information_requirement_scale_reduces_population() -> None:
    values = (0.5, 1.0, 2.0)
    populations = [
        _mean_metric(
            _run_payload(42, steps=40, reproduction_information_scale=value),
            "population",
        )
        for value in values
    ]

    assert populations[0] > populations[-1]
    assert populations[0] - populations[-1] >= 0.5


def test_attention_requirement_scale_reduces_population() -> None:
    values = (1.0, 1.5, 2.0)
    populations = [
        _mean_metric(
            _run_payload(42, steps=40, reproduction_attention_scale=value),
            "population",
        )
        for value in values
    ]

    assert populations[0] > populations[1] > populations[2]
    assert populations[0] - populations[-1] >= 1.0


def test_unrelated_reproduction_knob_does_not_change_event_locations() -> None:
    values = (0.0, 0.5, 1.0)
    event_location_counts = [
        {
            int(record["active_location_count"])
            for record in _records(
                _run_payload(42, steps=110, reproduction_coupling_strength=value)
            )
        }
        for value in values
    ]

    assert event_location_counts == [{0, 1}, {0, 1}, {0, 1}]
