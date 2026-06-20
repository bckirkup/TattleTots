#!/usr/bin/env python3
"""Diagnose why fire COP dispatch never fires."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from domain_runner.types import RunContext
from fire_ecology.runner import FireDomainHooks
from tattletots.engine.cop import fuse_reports_into_cops, select_dispatch_targets
from tattletots.integration.tattletots_layer import TattleTotsLayer

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE = _SCRIPT_DIR.parents[1]


def deep_merge(dict1: dict, dict2: dict) -> dict:
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def build_config() -> dict:
    saved = _SCRIPT_DIR / "quick_validation_results" / "fe_medium_fire_10drones_s42_config.json"
    if saved.exists():
        return json.loads(saved.read_text())

    base = json.loads(
        (_WORKSPACE / "Scrapiron_and_the_Bear/configs/tattletots_integration.json").read_text()
    )
    qv = json.loads((_WORKSPACE / "quick_validation_config.json").read_text())
    tt = qv["tattletots_config"].copy()
    tt.pop("seed", None)
    max_steps = tt.pop("max_steps", 800)
    tt["max_steps"] = max_steps
    tt["seed"] = 42
    scen = qv["fire_ecology"]["runs"][0]
    overrides = {
        "simulation": tt,
        "domain": {
            "steps": max_steps,
            "seed": 42,
            "n_drones": scen["n_drones"],
            "n_cameras": scen["n_cameras"],
            "base_ignition_rate": scen["base_ignition_rate"],
            "weather_volatility": scen["weather_volatility"],
        },
    }
    return deep_merge(base, overrides)


def main() -> int:
    cfg = build_config()
    domain_cfg = dict(cfg.get("domain", {}))
    steps = int(domain_cfg.pop("steps", 200))
    seed = int(domain_cfg.get("seed", 42))
    run = RunContext(
        steps=steps,
        seed=seed,
        domain_config=domain_cfg,
        layer="tattletots",
        simulation_config=dict(cfg.get("simulation", {})),
    )

    hooks = FireDomainHooks()
    adapter = hooks.build_adapter(run.domain_config)
    layer = TattleTotsLayer()
    state = layer.setup(adapter, run)
    world = state["world"]
    cops = state["cops"]
    sim = state["sim_config"]
    responder = adapter.get_responder_user_id()
    rcop = cops[responder]

    print(f"steps={run.steps} n_drones={adapter.n_drones}")
    print(
        f"cop threshold={sim.cop_dispatch_threshold} "
        f"min_reports={sim.cop_min_supporting_reports} "
        f"min_weight={sim.cop_min_supporting_weight}"
    )

    max_threat = 0.0
    max_above = 0
    total_targets = 0
    steps_with_reports = 0

    for step in range(run.steps):
        adapter.step(step)
        world.set_event_state(adapter.get_active_locations(step))
        record = world.step()
        if record.reports_issued:
            steps_with_reports += 1

        fuse_reports_into_cops(
            cops,
            world.last_reports,
            world.users,
            step,
            adapter=adapter,
            non_target_weight_scale=sim.cop_non_target_weight_scale,
        )
        targets = select_dispatch_targets(cops, responder, world.last_reports)
        summary = rcop.summary()
        max_threat = max(max_threat, float(summary["max_threat_level"]))
        max_above = max(max_above, int(summary["locations_above_threshold"]))
        total_targets += len(targets)

        if step < 5 or (record.reports_issued and step % 50 == 0):
            extra = ""
            if world.last_reports:
                r = world.last_reports[0]
                u = world.users[r.target_user_id]
                rel = u.compute_relevance(r.signal_vector)
                scale = (
                    1.0
                    if r.target_user_id == responder
                    else sim.cop_non_target_weight_scale
                )
                weight = u.get_trust(r.agent_id) * rel * r.confidence * scale
                contrib = weight * r.anomaly_score
                extra = (
                    f" w={weight:.4f} contrib={contrib:.4f} "
                    f"anom={r.anomaly_score:.3f} rel={rel:.3f} "
                    f"trust={u.get_trust(r.agent_id):.3f} "
                    f"to_responder={r.target_user_id == responder}"
                )
            print(
                f"step {step:4d}: reports={record.reports_issued} "
                f"correct={record.correct_reports} "
                f"max_threat={summary['max_threat_level']:.4f} "
                f"above={summary['locations_above_threshold']} "
                f"targets={len(targets)}{extra}"
            )

    print("--- summary ---")
    print(f"max_threat={max_threat:.4f} max_above={max_above} total_targets={total_targets}")
    print(f"steps_with_reports={steps_with_reports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
