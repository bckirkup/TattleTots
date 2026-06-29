"""Build designed-experiment run configurations."""

from __future__ import annotations

from typing import Any


def _append_run(
    runs: list[dict[str, Any]],
    *,
    name: str,
    domain: str,
    sim_config: dict[str, Any],
    domain_config: dict[str, Any],
) -> None:
    runs.append(
        {
            "name": name,
            "domain": domain,
            "sim_config": sim_config,
            "domain_config": domain_config,
        }
    )


def _build_coral_key_runs(
    domain_data: dict[str, Any],
    *,
    steps: int,
    seeds: list[int],
    tattletots_levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    factors = domain_data["factors"]
    for iuu in factors["iuu_vessel_count"]:
        for adv in factors["adversary_level"]:
            for sar in factors["sar_revisit_interval"]:
                for stream_dim in factors["stream_dimension"]:
                    for level_idx, tt_level in enumerate(tattletots_levels):
                        for seed in seeds:
                            run_name = (
                                f"ck_iuu{iuu}_adv{adv}_sar{sar}_dim{stream_dim}"
                                f"_ttL{level_idx + 1}_s{seed}"
                            )
                            sim_cfg = tt_level.copy()
                            sim_cfg["max_steps"] = steps
                            sim_cfg["seed"] = seed
                            adv_params = domain_data["adversary_levels"][adv]
                            dom_cfg = {
                                "total_epochs": steps,
                                "seed": seed,
                                "fleet": {
                                    "n_iuu_vessels": iuu,
                                    "underreport_fraction": adv_params["underreport_fraction"],
                                },
                                "sensors": {"sar_revisit_interval": sar},
                                "adversary": {
                                    "ais_disable_probability": adv_params["ais_disable_probability"],
                                    "spoof_probability": adv_params["spoof_probability"],
                                    "platform_interference_rate": adv_params[
                                        "platform_interference_rate"
                                    ],
                                },
                            }
                            _append_run(
                                runs,
                                name=run_name,
                                domain="coral_key",
                                sim_config=sim_cfg,
                                domain_config=dom_cfg,
                            )
    return runs


def _build_fire_ecology_runs(
    domain_data: dict[str, Any],
    *,
    steps: int,
    seeds: list[int],
    tattletots_levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    factors = domain_data["factors"]
    for phase in factors["deployment_phase"]:
        for dropout in factors["sensor_dropout"]:
            for stream_dim in factors["stream_dimension"]:
                for level_idx, tt_level in enumerate(tattletots_levels):
                    for seed in seeds:
                        run_name = (
                            f"fe_ph{phase}_drop{dropout.replace('%', '')}"
                            f"_dim{stream_dim}_ttL{level_idx + 1}_s{seed}"
                        )
                        sim_cfg = tt_level.copy()
                        sim_cfg["max_steps"] = steps
                        sim_cfg["seed"] = seed
                        phase_params = domain_data["phases"][phase].copy()
                        dropout_frac = float(dropout.replace("%", "")) / 100.0
                        phase_params["n_cameras"] = max(
                            0, int(phase_params["n_cameras"] * (1.0 - dropout_frac))
                        )
                        phase_params["n_weather_stations"] = max(
                            0, int(phase_params["n_weather_stations"] * (1.0 - dropout_frac))
                        )
                        phase_params["n_fuel_sensors"] = max(
                            0, int(phase_params["n_fuel_sensors"] * (1.0 - dropout_frac))
                        )
                        dom_cfg = {
                            "steps": steps,
                            "seed": seed,
                            "n_cameras": phase_params["n_cameras"],
                            "n_weather_stations": phase_params["n_weather_stations"],
                            "n_fuel_sensors": phase_params["n_fuel_sensors"],
                            "max_thermal_dim": stream_dim,
                        }
                        _append_run(
                            runs,
                            name=run_name,
                            domain="fire_ecology",
                            sim_config=sim_cfg,
                            domain_config=dom_cfg,
                        )
    return runs


def _build_grain_guard_runs(
    domain_data: dict[str, Any],
    *,
    steps: int,
    seeds: list[int],
    tattletots_levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    factors = domain_data["factors"]
    for landscape in factors["landscape"]:
        for budget in factors["sensor_budget"]:
            for stream_dim in factors["stream_dimension"]:
                for level_idx, tt_level in enumerate(tattletots_levels):
                    for seed in seeds:
                        run_name = (
                            f"gg_ls{landscape}_bud{budget}_dim{stream_dim}"
                            f"_ttL{level_idx + 1}_s{seed}"
                        )
                        sim_cfg = tt_level.copy()
                        sim_cfg["max_steps"] = steps
                        sim_cfg["seed"] = seed
                        budget_params = domain_data["sensor_budgets"][budget]
                        dom_cfg = {
                            "steps": steps,
                            "seed": seed,
                            "landscape": landscape,
                            "n_traps": budget_params["n_traps"],
                            "n_weather_stations": budget_params["n_weather_stations"],
                            "n_soil_sensors": budget_params["n_soil_sensors"],
                            "engine_max_dim": stream_dim,
                        }
                        _append_run(
                            runs,
                            name=run_name,
                            domain="grain_guard",
                            sim_config=sim_cfg,
                            domain_config=dom_cfg,
                        )
    return runs


def collect_experiment_runs(
    exp_config: dict[str, Any],
    *,
    domain_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Expand experiment config into executable run descriptors."""
    steps = exp_config["steps"]
    seeds = exp_config["seeds"]
    tattletots_levels = exp_config["tattletots_levels"]
    domains_data = exp_config["domains"]

    builders = {
        "coral_key": _build_coral_key_runs,
        "fire_ecology": _build_fire_ecology_runs,
        "grain_guard": _build_grain_guard_runs,
    }

    runs: list[dict[str, Any]] = []
    for domain_key, domain_data in domains_data.items():
        if domain_filter and domain_key != domain_filter:
            continue
        builder = builders[domain_key]
        runs.extend(
            builder(
                domain_data,
                steps=steps,
                seeds=seeds,
                tattletots_levels=tattletots_levels,
            )
        )
    return runs
