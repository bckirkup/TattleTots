"""End-to-end guards for seeded simulation determinism."""

from __future__ import annotations

import hashlib
import io
import json
import math
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
FLOAT_REFERENCE_PATH = REPO_ROOT / "tests" / "data" / "determinism_float_references.json"
# CHANGE DETECTOR, not a correctness check. This hash covers only the
# float-free structure of both seeded runs; it does not validate float telemetry.
EXPECTED_STRUCTURAL_FINGERPRINT = "aa1231053791820c27bc599370e56b51bf2148c1dc30e1550528344cb49207da"


def _canonicalize(value: object) -> object:
    """Normalize floating-point serialization for within-process fingerprints."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    return value


def _structural_projection(value: object) -> object:
    """Keep the payload structure while omitting floating-point leaves."""
    if isinstance(value, dict):
        return {
            key: _structural_projection(item)
            for key, item in value.items()
            if not isinstance(item, float)
        }
    if isinstance(value, list):
        return [_structural_projection(item) for item in value if not isinstance(item, float)]
    if isinstance(value, tuple):
        return tuple(_structural_projection(item) for item in value if not isinstance(item, float))
    return value


def _structural_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float_fields(record: dict[str, object]) -> dict[str, float]:
    return {key: value for key, value in record.items() if isinstance(value, float)}


def _load_float_references() -> dict[str, list[dict[str, float]]]:
    return json.loads(FLOAT_REFERENCE_PATH.read_text(encoding="utf-8"))


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


def _assert_float_telemetry_bounds(payload: dict[str, object]) -> None:
    records = payload["records"]
    assert isinstance(records, list)
    for index, record in enumerate(records):
        assert isinstance(record, dict)
        for field, value in record.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"record[{index}].{field} is not finite"
                if field.endswith("_share"):
                    assert 0.0 <= value <= 1.0, f"record[{index}].{field} is out of bounds"
                else:
                    assert value >= 0.0, f"record[{index}].{field} is negative"
            elif isinstance(value, int) and not isinstance(value, bool):
                assert value >= 0, f"record[{index}].{field} is negative"


def _assert_float_telemetry(
    label: str,
    payload: dict[str, object],
    references: list[dict[str, float]],
) -> None:
    records = payload["records"]
    assert isinstance(records, list)
    assert len(records) == len(references), f"{label} record count changed"
    for index, (record, reference) in enumerate(zip(records, references, strict=True)):
        assert isinstance(record, dict)
        actual_fields = _float_fields(record)
        if set(actual_fields) != set(reference):
            pytest.fail(
                f"{label} record[{index}] float fields changed: "
                f"expected={sorted(reference)} actual={sorted(actual_fields)}"
            )
        for field, expected in reference.items():
            actual = actual_fields[field]
            if actual != pytest.approx(expected, rel=1e-9):
                relative_difference = (
                    abs(actual - expected) / abs(expected) if expected else abs(actual)
                )
                pytest.fail(
                    f"{label} record[{index}].{field} differs: "
                    f"expected={expected!r} actual={actual!r} "
                    f"relative difference={relative_difference:.17g}\n"
                    f"{_environment_diagnostics()}"
                )


def test_seeded_runs_have_structural_golden_change_detector() -> None:
    default_payload = _run_payload(42)
    legacy_payload = _run_payload(
        42,
        reproduction_coupling_strength=0.0,
        grounding_quality_strength=0.0,
    )
    projection = {
        "default": _structural_projection(default_payload),
        "legacy": _structural_projection(legacy_payload),
    }
    assert _structural_fingerprint(projection) == EXPECTED_STRUCTURAL_FINGERPRINT


def test_float_telemetry_matches_both_run_references() -> None:
    payloads = {
        "default": _run_payload(42),
        "legacy": _run_payload(
            42,
            reproduction_coupling_strength=0.0,
            grounding_quality_strength=0.0,
        ),
    }
    references = _load_float_references()
    for label, payload in payloads.items():
        _assert_float_telemetry_bounds(payload)
        _assert_float_telemetry(label, payload, references[label])


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
