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
STRUCTURAL_REFERENCE_PATH = REPO_ROOT / "tests" / "data" / "determinism_structural_references.json"
AGENT_TRACE_REFERENCE_PATH = REPO_ROOT / "tests" / "data" / "determinism_agent_traces.json"
# CHANGE DETECTOR, not a correctness check. These references cover only the
# float-free structure of both seeded runs; they do not validate float telemetry.
# The agent trace is tolerance-based reference data for locating divergence,
# not a correctness check or a replacement for the structural change detector.


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
        return [_structural_projection(item) for item in value if not isinstance(item, float)]
    return value


def _load_structural_references() -> dict[str, object]:
    return json.loads(STRUCTURAL_REFERENCE_PATH.read_text(encoding="utf-8"))


def _structural_differences(
    expected: object,
    actual: object,
    path: str,
) -> list[tuple[str, object, object]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path} {key}" if path in {"default", "legacy"} else f"{path}.{key}"
            if key not in expected:
                differences.append((child_path, "<missing>", actual[key]))
            elif key not in actual:
                differences.append((child_path, expected[key], "<missing>"))
            else:
                differences.extend(_structural_differences(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            child_path = f"{path}[{index}]"
            if index >= len(expected):
                differences.append((child_path, "<missing>", actual[index]))
            elif index >= len(actual):
                differences.append((child_path, expected[index], "<missing>"))
            else:
                differences.extend(
                    _structural_differences(expected[index], actual[index], child_path)
                )
        return differences
    if expected != actual:
        return [(path, expected, actual)]
    return []


def _float_fields(record: dict[str, object]) -> dict[str, float]:
    return {key: value for key, value in record.items() if isinstance(value, float)}


def _load_float_references() -> dict[str, list[dict[str, float]]]:
    return json.loads(FLOAT_REFERENCE_PATH.read_text(encoding="utf-8"))


def _load_agent_trace_references() -> dict[str, list[dict[str, object]]]:
    return json.loads(AGENT_TRACE_REFERENCE_PATH.read_text(encoding="utf-8"))


def _capture_agent_trace(world: World) -> dict[str, dict[str, float]]:
    return {
        agent_id: {
            "attention_energy": world.agents[agent_id].state.energy.attention,
            "information_energy": world.agents[agent_id].state.energy.information,
            "attention_income": world.agents[agent_id].state.last_step_attention_income,
            "attention_delta": world._attention_deltas.get(agent_id, 0.0),
        }
        for agent_id in sorted(world.agents)
    }


def _run_payload(
    seed: int,
    steps: int = 8,
    *,
    reproduction_coupling_strength: float = 1.0,
    reproduction_information_scale: float = 1.0,
    reproduction_attention_scale: float = 1.0,
    grounding_quality_strength: float = 0.5,
    capture_agent_trace: bool = False,
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

    agent_trace: list[dict[str, object]] = []
    for step in range(steps):
        scenario.step(step)
        world.set_event_state(scenario.get_active_locations(step))
        world.step()
        if capture_agent_trace:
            agent_trace.append(
                {
                    "time_step": world.time_step,
                    "agents": _capture_agent_trace(world),
                }
            )

    payload: dict[str, object] = {
        "agents": sorted(world.agents),
        "streams": sorted(world.streams),
        "users": sorted(world.users),
        "records": [asdict(record) for record in world.telemetry.history],
    }
    if capture_agent_trace:
        payload["agent_trace"] = agent_trace
    return payload


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
    runtime = io.StringIO()
    with redirect_stdout(runtime):
        if hasattr(np, "show_runtime"):
            np.show_runtime()
    return "\n".join(
        [
            f"python: {platform.python_version()} ({sys.executable})",
            f"numpy: {np.__version__}",
            f"scipy: {scipy.__version__}",
            "numpy.show_config():",
            config.getvalue().rstrip(),
            "numpy.show_runtime():",
            runtime.getvalue().rstrip() or "unavailable",
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
    differences: list[tuple[float, str, float, float]] = []
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
                differences.append(
                    (
                        relative_difference,
                        f"{label} records[{index}].{field}",
                        expected,
                        actual,
                    )
                )
    if differences:
        differences.sort(reverse=True)
        lines = [
            f"float telemetry differences: showing {min(20, len(differences))} "
            f"of {len(differences)}",
        ]
        lines.extend(
            f"{path}: expected={expected!r} actual={actual!r} "
            f"relative difference={relative_difference:.17g}"
            for relative_difference, path, expected, actual in differences[:20]
        )
        pytest.fail("\n".join([*lines, _environment_diagnostics()]))


def _agent_trace_differences(
    expected: list[dict[str, object]],
    actual: list[dict[str, object]],
) -> list[tuple[int, str, str, object, object, float]]:
    fields = (
        "attention_energy",
        "information_energy",
        "attention_income",
        "attention_delta",
    )
    differences: list[tuple[int, str, str, object, object, float]] = []
    for index in range(max(len(expected), len(actual))):
        expected_step = expected[index] if index < len(expected) else None
        actual_step = actual[index] if index < len(actual) else None
        step = index + 1
        if expected_step is None or actual_step is None:
            differences.append(
                (
                    step,
                    "<step>",
                    "step_presence",
                    expected_step if expected_step is not None else "<missing>",
                    actual_step if actual_step is not None else "<missing>",
                    math.inf,
                )
            )
            continue
        expected_step_number = expected_step.get("time_step", step)
        actual_step_number = actual_step.get("time_step", step)
        if expected_step_number != actual_step_number:
            differences.append(
                (step, "<step>", "time_step", expected_step_number, actual_step_number, math.inf)
            )
        expected_agents = expected_step.get("agents", {})
        actual_agents = actual_step.get("agents", {})
        if not isinstance(expected_agents, dict) or not isinstance(actual_agents, dict):
            differences.append(
                (step, "<step>", "agent_trace_shape", expected_agents, actual_agents, math.inf)
            )
            continue
        for agent_id in sorted(set(expected_agents) | set(actual_agents)):
            expected_agent = expected_agents.get(agent_id)
            actual_agent = actual_agents.get(agent_id)
            if not isinstance(expected_agent, dict) or not isinstance(actual_agent, dict):
                differences.append(
                    (
                        step,
                        agent_id,
                        "agent_presence",
                        "present" if isinstance(expected_agent, dict) else "<missing>",
                        "present" if isinstance(actual_agent, dict) else "<missing>",
                        math.inf,
                    )
                )
                continue
            for field in fields:
                expected_value = expected_agent[field]
                actual_value = actual_agent[field]
                if not isinstance(expected_value, float) or not isinstance(actual_value, float):
                    if expected_value != actual_value:
                        differences.append(
                            (step, agent_id, field, expected_value, actual_value, math.inf)
                        )
                    continue
                if actual_value != pytest.approx(expected_value, rel=1e-9):
                    relative_difference = (
                        abs(actual_value - expected_value) / abs(expected_value)
                        if expected_value
                        else abs(actual_value)
                    )
                    differences.append(
                        (
                            step,
                            agent_id,
                            field,
                            expected_value,
                            actual_value,
                            relative_difference,
                        )
                    )
    return differences


def _assert_agent_trace(
    label: str,
    actual_payload: dict[str, object],
    references: list[dict[str, object]],
) -> None:
    actual_trace = actual_payload["agent_trace"]
    assert isinstance(actual_trace, list)
    differences = _agent_trace_differences(references, actual_trace)
    if not differences:
        return
    first_step, first_agent, first_field, first_expected, first_actual, first_relative = (
        differences[0]
    )
    lines = [
        f"first agent-trace divergence: {label} step {first_step}, "
        f"agent {first_agent}, field {first_field}, "
        f"expected={first_expected!r} actual={first_actual!r} "
        f"relative difference={first_relative:.17g}",
        "",
        "agent-trace divergence growth:",
    ]
    by_step: dict[int, list[tuple[int, str, str, object, object, float]]] = {}
    for difference in differences:
        by_step.setdefault(difference[0], []).append(difference)
    for step, step_differences in sorted(by_step.items()):
        worst_relative = max(difference[-1] for difference in step_differences)
        lines.append(
            f"step {step}: {len(step_differences)} fields, worst rel={worst_relative:.17g}"
        )
    lines.extend(
        [
            "",
            f"worst agent-trace differences: showing {min(20, len(differences))} "
            f"of {len(differences)}",
        ]
    )
    for step, agent_id, field, expected, actual, relative_difference in sorted(
        differences, key=lambda difference: difference[-1], reverse=True
    )[:20]:
        lines.append(
            f"{label} step {step} agent {agent_id} {field}: "
            f"expected={expected!r} actual={actual!r} "
            f"relative difference={relative_difference:.17g}"
        )
    pytest.fail("\n".join([*lines, _environment_diagnostics()]))


def _step_summary(record: object) -> str:
    if not isinstance(record, dict):
        return "invalid"
    return (
        f"time_step={record.get('time_step')}, "
        f"population={record.get('population')}, "
        f"n_adults={record.get('n_adults')}, "
        f"n_streams={record.get('n_streams')}"
    )


def _structural_diagnostics(
    expected: dict[str, object],
    actual_payloads: dict[str, dict[str, object]],
) -> str:
    actual = {label: _structural_projection(payload) for label, payload in actual_payloads.items()}
    differences = [
        difference
        for label in ("default", "legacy")
        for difference in _structural_differences(expected[label], actual[label], label)
    ]
    if not differences:
        return ""
    lines = [
        f"structural differences: showing {min(40, len(differences))} of {len(differences)}",
    ]
    lines.extend(
        f"{path}: expected={expected_value!r} actual={actual_value!r}"
        for path, expected_value, actual_value in differences[:40]
    )
    lines.extend(
        [
            "",
            "id-list lengths:",
            "run      | expected agents/streams/users | actual agents/streams/users",
        ]
    )
    for label in ("default", "legacy"):
        expected_run = expected[label]
        actual_run = actual[label]
        expected_lengths = "/".join(
            str(len(expected_run[key])) for key in ("agents", "streams", "users")
        )
        actual_lengths = "/".join(
            str(len(actual_run[key])) for key in ("agents", "streams", "users")
        )
        lines.append(f"{label:<8} | {expected_lengths:<29} | {actual_lengths}")
    lines.extend(
        [
            "",
            "per-step records:",
            "run      step | expected time_step/population/n_adults/n_streams "
            "| actual time_step/population/n_adults/n_streams",
        ]
    )
    for label in ("default", "legacy"):
        expected_records = expected[label]["records"]
        actual_records = actual_payloads[label]["records"]
        for index in range(max(len(expected_records), len(actual_records))):
            expected_record = expected_records[index] if index < len(expected_records) else None
            actual_record = actual_records[index] if index < len(actual_records) else None
            lines.append(
                f"{label:<8} {index:>4} | {_step_summary(expected_record)} "
                f"| {_step_summary(actual_record)}"
            )
    return "\n".join(lines)


def test_seeded_runs_have_structural_golden_change_detector() -> None:
    payloads = {
        "default": _run_payload(42),
        "legacy": _run_payload(
            42,
            reproduction_coupling_strength=0.0,
            grounding_quality_strength=0.0,
        ),
    }
    references = _load_structural_references()
    diagnostics = _structural_diagnostics(references, payloads)
    if diagnostics:
        pytest.fail("\n".join([diagnostics, _environment_diagnostics()]))


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


def test_seeded_agent_traces_match_references() -> None:
    payloads = {
        "default": _run_payload(42, capture_agent_trace=True),
        "legacy": _run_payload(
            42,
            reproduction_coupling_strength=0.0,
            grounding_quality_strength=0.0,
            capture_agent_trace=True,
        ),
    }
    references = _load_agent_trace_references()
    for label, payload in payloads.items():
        _assert_agent_trace(label, payload, references[label])


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
