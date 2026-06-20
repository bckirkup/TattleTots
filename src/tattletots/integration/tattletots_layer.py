"""TattleTots orchestration layer — optional plugin for domain simulations."""

from __future__ import annotations

from typing import Any

from tattletots.engine.config import SimulationConfig
from tattletots.engine.dispatch_integration import init_user_cops, run_dispatch_cycle
from tattletots.engine.relevance import align_user_priorities_to_report_space
from tattletots.engine.response_judgment import aggregate_outcomes
from tattletots.engine.world import World
from tattletots.output_schema import (
    CostMetrics,
    EcologyMetrics,
    RunSummary,
    SimulationOutput,
    TimeSeries,
)
from tattletots.telemetry.cost_accounting import CostAccumulator

try:
    from domain_runner.layer import SimulationLayer
    from domain_runner.types import RunContext
except ImportError as exc:
    raise ImportError(
        "TattleTots integration layer requires the domain-runner package"
    ) from exc


class TattleTotsLayer:
    """Agent ecology + COP dispatch layer above a domain adapter."""

    name = "tattletots"

    def setup(self, adapter: Any, run: RunContext) -> dict[str, Any]:
        sim_config = SimulationConfig(**run.simulation_config)
        world = World(config=sim_config)
        for stream in adapter.get_streams():
            world.add_stream(stream)
        for user in adapter.get_users():
            world.add_user(user)
        world.set_location_inference(adapter.infer_report_location)
        world.seed_population()
        align_user_priorities_to_report_space(world)
        cops = init_user_cops(world, adapter, sim_config)
        return {
            "world": world,
            "sim_config": sim_config,
            "cops": cops,
            "cost_accumulator": CostAccumulator(),
            "steps_completed": 0,
        }

    def step(self, adapter: Any, step: int, layer_state: dict[str, Any]) -> dict[str, Any]:
        world: World = layer_state["world"]
        sim_config: SimulationConfig = layer_state["sim_config"]
        cops = layer_state["cops"]

        adapter.step(step)
        world.set_event_state(adapter.get_active_locations(step))
        record = world.step()
        outcomes, _ = run_dispatch_cycle(world, adapter, cops, step, sim_config)
        outcome_counts = aggregate_outcomes(outcomes)

        cost_dict = adapter.compute_costs(
            n_escalations=record.reports_issued,
            n_correct=record.correct_reports,
            n_false_alarms=record.false_alarms,
            n_missed=record.missed_events,
        )
        layer_state["cost_accumulator"].record_from_dict(record.time_step, cost_dict)
        layer_state["steps_completed"] = step + 1
        layer_state["last_record"] = record
        layer_state["last_outcomes"] = outcomes
        layer_state["last_outcome_counts"] = outcome_counts
        layer_state["last_cost_dict"] = cost_dict

        return {
            "population": record.population,
            "reports_issued": record.reports_issued,
            "correct_reports": record.correct_reports,
            "false_alarms": record.false_alarms,
            "missed_events": record.missed_events,
            "outcome_counts": outcome_counts,
            "cost_dict": cost_dict,
            "stop": record.population == 0,
        }

    def finalize(
        self,
        adapter: Any,
        layer_state: dict[str, Any],
        run: RunContext,
    ) -> dict[str, Any]:
        world: World = layer_state["world"]
        sim_config: SimulationConfig = layer_state["sim_config"]
        cost_accumulator: CostAccumulator = layer_state["cost_accumulator"]

        summary = world.telemetry.summary()
        cost_summary = cost_accumulator.summary()

        output = SimulationOutput(
            run_summary=RunSummary(
                domain=adapter.__class__.__name__,
                steps_completed=world.telemetry.total_steps,
                seed=run.seed,
                wall_time_seconds=0.0,
            ),
            simulation_config=sim_config.model_dump(),
            domain_config=run.domain_config,
            ecology_metrics=EcologyMetrics(
                final_population=int(summary["final_population"]),
                peak_population=int(summary["peak_population"]),
                total_births=int(summary["total_births"]),
                total_deaths=int(summary["total_deaths"]),
                total_reports=int(summary["total_reports"]),
                precision=float(summary["precision"]),
                max_trophic_depth=float(summary["max_trophic_depth"]),
                reached_equilibrium=bool(summary["reached_equilibrium"]),
                total_responses_dispatched=int(summary["total_responses_dispatched"]),
                total_responses_judged_necessary=int(summary["total_responses_judged_necessary"]),
                total_responses_judged_unnecessary=int(summary["total_responses_judged_unnecessary"]),
                responder_necessity_rate=float(summary["responder_necessity_rate"]),
                unnecessary_dispatch_rate=float(summary["unnecessary_dispatch_rate"]),
            ),
            cost_metrics=CostMetrics(
                total_surveillance_cost=cost_summary["total_surveillance_cost"],
                total_response_cost=cost_summary["total_response_cost"],
                total_damage_cost=cost_summary["total_damage_cost"],
                total_cost=cost_summary["total_cost"],
                mean_cost_per_step=cost_summary["mean_cost_per_step"],
            ),
            domain_metrics={},
            time_series=TimeSeries.from_telemetry(
                world.telemetry, cost_accumulator.cost_history()
            ),
        )

        return {
            "telemetry_summary": summary,
            "cost_summary": cost_summary,
            "simulation_output": output,
            "world": world,
        }


def resolve_layer(name: str) -> SimulationLayer:
    """Resolve a layer name to an orchestration layer instance."""
    from domain_runner.layer import DomainOnlyLayer

    if name in ("domain_only", "domain", "none"):
        return DomainOnlyLayer()
    if name in ("tattletots", "tots"):
        return TattleTotsLayer()
    raise ValueError(
        f"Unknown simulation layer {name!r}. "
        "Supported: domain_only, tattletots"
    )
