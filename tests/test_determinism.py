"""End-to-end guards for seeded simulation determinism."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import scipy

from tattletots.cli import _load_scenario
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.models.identity import stable_id_digest
from tattletots.scenarios.gaussian_shift import GaussianShiftScenario

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FINGERPRINT = "4b98dbd41feb9fcd0d207de2b9a7b54fedacfd6f6bf8ac99b780e2829fcad8bc"
EXPECTED_LEGACY_FINGERPRINT = "5263ba65680de33c6fe1d6dd693e725a084bd6e0f5fbe64fc06678d299c14b23"


def _canonicalize(value: object) -> object:
    """Normalize floating-point serialization across supported runtimes."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    return value


def _run_payload(
    seed: int,
    steps: int = 8,
    *,
    reproduction_coupling_strength: float = 1.0,
    reproduction_information_scale: float = 1.0,
    reproduction_attention_scale: float = 1.0,
    grounding_quality_strength: float = 0.5,
    attention_trace: dict[int, dict[str, float]] | None = None,
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
        if attention_trace is not None:
            attention_trace[step + 1] = dict(world._attention_deltas)

    return {
        "agents": sorted(world.agents),
        "streams": sorted(world.streams),
        "users": sorted(world.users),
        "records": [asdict(record) for record in world.telemetry.history],
    }


def _run_fingerprint(
    seed: int,
    steps: int = 8,
    *,
    reproduction_coupling_strength: float = 1.0,
    reproduction_information_scale: float = 1.0,
    reproduction_attention_scale: float = 1.0,
    grounding_quality_strength: float = 0.5,
) -> str:
    payload = _run_payload(
        seed,
        steps,
        reproduction_coupling_strength=reproduction_coupling_strength,
        reproduction_information_scale=reproduction_information_scale,
        reproduction_attention_scale=reproduction_attention_scale,
        grounding_quality_strength=grounding_quality_strength,
    )
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_diagnostics() -> str:
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    return "\n".join(
        [
            f"python: {platform.python_version()} ({sys.executable})",
            f"numpy: {np.__version__}",
            f"scipy: {scipy.__version__}",
            "numpy.show_config():",
            config.getvalue().rstrip(),
        ]
    )


def _payload_diagnostics(payload: dict[str, object]) -> str:
    records = payload["records"]
    assert isinstance(records, list)
    lines = []
    for section in ("agents", "streams", "users"):
        value = payload[section]
        encoded = json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":")).encode()
        lines.append(f"{section} digest={hashlib.sha256(encoded).hexdigest()}")
    for index, record in enumerate(records):
        canonical = _canonicalize(record)
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        values = _float_values(record)
        magnitude = max((abs(value) for value in values), default=0.0)
        lines.append(
            f"record[{index}] max_abs={magnitude:.17g} digest={hashlib.sha256(encoded).hexdigest()}"
        )
        assert isinstance(record, dict)
        for key in sorted(record):
            value = _canonicalize(record[key])
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            lines.append(
                f"record[{index}].{key}={value!r} digest={hashlib.sha256(encoded).hexdigest()}"
            )
    return "\n".join(lines)


def _float_values(value: object) -> list[float]:
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _float_values(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _float_values(child)]
    return []


def _assert_fingerprint(
    expected: str,
    payload: dict[str, object],
    attention_trace: dict[int, dict[str, float]] | None = None,
) -> None:
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected:
        boundary_lines = []
        if attention_trace is not None and 6 in attention_trace:
            boundary_lines = [
                "attention deltas at step 6:",
                *(
                    f"{agent_id}={delta:.17g}"
                    for agent_id, delta in sorted(attention_trace[6].items())
                ),
            ]
        pytest.fail(
            "\n".join(
                [
                    f"fingerprint mismatch: expected={expected} actual={actual}",
                    _environment_diagnostics(),
                    "per-record diagnostics:",
                    _payload_diagnostics(payload),
                    *boundary_lines,
                ]
            )
        )


def test_seeded_run_has_golden_fingerprint() -> None:
    _assert_fingerprint(EXPECTED_FINGERPRINT, _run_payload(42))


def test_legacy_setting_has_branch_golden_fingerprint() -> None:
    attention_trace: dict[int, dict[str, float]] = {}
    payload = _run_payload(
        42,
        reproduction_coupling_strength=0.0,
        grounding_quality_strength=0.0,
        attention_trace=attention_trace,
    )

    _assert_fingerprint(EXPECTED_LEGACY_FINGERPRINT, payload, attention_trace)


def test_reproduction_coupling_config_changes_run_fingerprint() -> None:
    legacy = _run_fingerprint(42, reproduction_coupling_strength=0.0)
    coupled = _run_fingerprint(42, reproduction_coupling_strength=1.0)

    assert legacy != coupled


def test_grounding_quality_config_changes_run_fingerprint() -> None:
    legacy = _run_fingerprint(42, grounding_quality_strength=0.0)
    coupled = _run_fingerprint(42, grounding_quality_strength=0.5)

    assert legacy != coupled


def test_information_requirement_scale_changes_run_fingerprint() -> None:
    baseline = _run_fingerprint(42, steps=40)
    scaled = _run_fingerprint(42, steps=40, reproduction_information_scale=2.0)

    assert baseline != scaled


def test_attention_requirement_scale_changes_run_fingerprint() -> None:
    baseline = _run_fingerprint(42, steps=40)
    scaled = _run_fingerprint(42, steps=40, reproduction_attention_scale=2.0)

    assert baseline != scaled


def test_same_seed_reproduces_the_complete_run() -> None:
    first = _run_fingerprint(42)
    second = _run_fingerprint(42)

    assert first == second


def test_same_seed_has_zero_fingerprint_spread_across_repeated_runs() -> None:
    fingerprints = {_run_fingerprint(42) for _ in range(10)}

    assert len(fingerprints) == 1


def test_different_seed_changes_the_complete_run() -> None:
    assert _run_fingerprint(42) != _run_fingerprint(43)


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


def canonicalize(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize(item) for item in value)
    return value

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
canonical = canonicalize(payload)
print(hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
"""
    fingerprints: list[str] = []
    for hash_seed in ("0", "1"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        fingerprints.append(result.stdout.strip())

    assert fingerprints[0] == fingerprints[1]
