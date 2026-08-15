#!/usr/bin/env python3
"""Why does an agent only issue ~0.46 reports in its lifetime?

`docs/heritability-measurement.md` located break 3 in reporting opportunity per
lifetime: the genome controls precision (clone intraclass correlation 0.63) but an
adult lives ~6.8 adult steps and reports on ~6.8% of them, so its own precision is
97% binomial noise and selection has no per-individual signal. Lifetime reports are
the product of two throttles, and this script measures both without changing the
engine.

Throttle A -- how long an agent stays alive and adult:
  * age at death, adult steps, and the juvenile share of a life;
  * which currency ran out (an agent dies when information OR attention hits zero);
  * the per-adult-step drift of each reserve, which says whether maintenance outruns
    income and in which currency.

Throttle B -- the per-adult-step reporting funnel:
  * share of adult steps with any grounded (raw-domain) yield -- is evidence arriving;
  * share where the normalized anomaly cleared the effective escalation threshold;
  * share that actually escalated;
  * the anomaly/threshold gap, which says whether the threshold or the signal is binding.

Prints only; it writes no artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import measurement_support
import numpy as np

from tattletots.models.agent import Agent, LifecycleStage

harness = measurement_support.load_harness()


@dataclass
class LifeRecord:
    """One agent's life, accumulated from the states observed each step."""

    agent_id: str
    adult_steps: int = 0
    juvenile_steps: int = 0
    reports: int = 0
    escalations: int = 0
    steps_with_grounded_yield: int = 0
    steps_above_threshold: int = 0
    anomaly_gaps: list[float] = field(default_factory=list)
    attention_deltas: list[float] = field(default_factory=list)
    information_deltas: list[float] = field(default_factory=list)
    last_attention: float = 0.0
    last_information: float = 0.0
    died: bool = False
    death_currency: str = ""


class OpportunityLedger:
    """Accumulates per-adult-step observables and mortality causes."""

    def __init__(self) -> None:
        self._records: dict[str, LifeRecord] = {}

    @property
    def records(self) -> list[LifeRecord]:
        """Snapshot of every agent observed."""
        return list(self._records.values())

    def observe(self, world: Any) -> None:
        """Fold one completed step of every agent's state into its life record."""
        for agent in world.agents.values():
            record = self._records.get(agent.id)
            if record is None:
                record = LifeRecord(
                    agent_id=agent.id,
                    last_attention=agent.state.energy.attention,
                    last_information=agent.state.energy.information,
                )
                self._records[agent.id] = record
            self._observe_agent(record, agent)

    def _observe_agent(self, record: LifeRecord, agent: Agent) -> None:
        # Insolvent agents keep their lifecycle stage until the step loop purges them,
        # and the loop only visits agents it already considered alive, so liveness must
        # be read from `is_alive` rather than from the stage.
        if not agent.is_alive:
            if not record.died:
                record.died = True
                record.death_currency = _exhausted_currency(agent)
            return
        stage = agent.state.lifecycle
        record.attention_deltas.append(agent.state.energy.attention - record.last_attention)
        record.information_deltas.append(agent.state.energy.information - record.last_information)
        record.last_attention = agent.state.energy.attention
        record.last_information = agent.state.energy.information
        if stage == LifecycleStage.JUVENILE:
            record.juvenile_steps += 1
            return
        record.adult_steps += 1
        if agent.state.last_escalated:
            record.escalations += 1
        if agent.state.last_step_grounded_yield > 0.0:
            record.steps_with_grounded_yield += 1
        gap = agent.state.last_anomaly_score - agent.state.effective_escalation_threshold
        record.anomaly_gaps.append(gap)
        if gap >= 0.0:
            record.steps_above_threshold += 1

    def finalize(self, world: Any) -> None:
        """Record final report counts, which are cumulative on agent state."""
        for agent in world.agents.values():
            record = self._records.get(agent.id)
            if record is not None:
                record.reports = agent.state.reports_issued


def _exhausted_currency(agent: Agent) -> str:
    """Which reserve was non-positive when the agent died."""
    information_gone = agent.state.energy.information <= 0.0
    attention_gone = agent.state.energy.attention <= 0.0
    if information_gone and attention_gone:
        return "both"
    if information_gone:
        return "information"
    if attention_gone:
        return "attention"
    return "neither"


def _share(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def lifespan_metrics(records: Sequence[LifeRecord]) -> dict[str, float]:
    """Throttle A: how long agents stay alive, and which currency ends them."""
    adults = [record for record in records if record.adult_steps > 0]
    if not adults:
        return {"n_adults": 0.0}
    adult_steps = np.array([float(record.adult_steps) for record in adults])
    juvenile_steps = np.array([float(record.juvenile_steps) for record in adults])
    deaths = [record for record in records if record.died]
    attention_drift = np.array(
        [float(np.mean(record.attention_deltas)) for record in adults if record.attention_deltas]
    )
    information_drift = np.array(
        [
            float(np.mean(record.information_deltas))
            for record in adults
            if record.information_deltas
        ]
    )
    return {
        "n_adults": float(adult_steps.size),
        "mean_adult_steps": float(adult_steps.mean()),
        "median_adult_steps": float(np.median(adult_steps)),
        "p90_adult_steps": float(np.percentile(adult_steps, 90)),
        "mean_juvenile_steps": float(juvenile_steps.mean()),
        "juvenile_share_of_life": float(
            juvenile_steps.sum() / (juvenile_steps + adult_steps).sum()
        ),
        "death_share_of_agents": _share(len(deaths), len(records)),
        "deaths_by_information": _share(
            sum(1 for record in deaths if record.death_currency == "information"), len(deaths)
        ),
        "deaths_by_attention": _share(
            sum(1 for record in deaths if record.death_currency == "attention"), len(deaths)
        ),
        "deaths_by_both": _share(
            sum(1 for record in deaths if record.death_currency == "both"), len(deaths)
        ),
        "mean_attention_drift_per_step": float(attention_drift.mean()),
        "mean_information_drift_per_step": float(information_drift.mean()),
        "share_with_negative_attention_drift": float((attention_drift < 0.0).mean()),
        "share_with_negative_information_drift": float((information_drift < 0.0).mean()),
    }


def funnel_metrics(records: Sequence[LifeRecord]) -> dict[str, float]:
    """Throttle B: the per-adult-step path from evidence to an issued report."""
    adults = [record for record in records if record.adult_steps > 0]
    if not adults:
        return {"n_adults": 0.0}
    adult_steps = sum(record.adult_steps for record in adults)
    gaps = np.concatenate(
        [np.asarray(record.anomaly_gaps) for record in adults if record.anomaly_gaps]
    )
    return {
        "adult_steps": float(adult_steps),
        "share_adult_steps_with_grounded_yield": _share(
            sum(record.steps_with_grounded_yield for record in adults), adult_steps
        ),
        "share_adult_steps_above_threshold": _share(
            sum(record.steps_above_threshold for record in adults), adult_steps
        ),
        "share_adult_steps_escalated": _share(
            sum(record.escalations for record in adults), adult_steps
        ),
        "reports_per_adult_step": _share(sum(record.reports for record in adults), adult_steps),
        "median_anomaly_minus_threshold": float(np.median(gaps)),
        "p90_anomaly_minus_threshold": float(np.percentile(gaps, 90)),
    }


def silence_survival_coupling(records: Sequence[LifeRecord]) -> dict[str, float]:
    """Does reporting buy the attention that keeps an adult alive?

    Almost every death is attention insolvency and attention income is only paid on
    reports, so the two throttles may be one loop: a silent agent starves, and a
    starved agent has no time to report.
    """
    adults = [record for record in records if record.adult_steps >= 2]
    if len(adults) < 3:
        return {"n_adults": float(len(adults))}
    escalation_rate = np.array(
        [record.escalations / record.adult_steps for record in adults], dtype=np.float64
    )
    lifespan = np.array([float(record.adult_steps) for record in adults], dtype=np.float64)
    attention_drift = np.array(
        [float(np.mean(record.attention_deltas)) for record in adults], dtype=np.float64
    )
    silent = escalation_rate <= 0.0
    metrics = {
        "n_adults": float(len(adults)),
        "silent_share_of_adults": float(silent.mean()),
        "mean_adult_steps_silent": float(lifespan[silent].mean()) if silent.any() else 0.0,
        "mean_adult_steps_reporting": (float(lifespan[~silent].mean()) if (~silent).any() else 0.0),
        "mean_attention_drift_silent": (
            float(attention_drift[silent].mean()) if silent.any() else 0.0
        ),
        "mean_attention_drift_reporting": (
            float(attention_drift[~silent].mean()) if (~silent).any() else 0.0
        ),
    }
    if escalation_rate.std() > 1e-12 and lifespan.std() > 1e-12:
        metrics["corr_escalation_rate_adult_steps"] = float(
            np.corrcoef(escalation_rate, lifespan)[0, 1]
        )
    if escalation_rate.std() > 1e-12 and attention_drift.std() > 1e-12:
        metrics["corr_escalation_rate_attention_drift"] = float(
            np.corrcoef(escalation_rate, attention_drift)[0, 1]
        )
    return metrics


def counterfactual_lifetime_reports(
    lifespan: dict[str, float],
    funnel: dict[str, float],
    target_reports: float,
) -> dict[str, float]:
    """Adult steps or escalation rate each throttle would need on its own.

    Break 3 needs ~7 reports per agent for an individual's precision to carry half
    its genome's signal, so this reports what each throttle would have to deliver
    alone to get there.
    """
    rate = funnel.get("reports_per_adult_step", 0.0)
    steps = lifespan.get("mean_adult_steps", 0.0)
    return {
        "target_reports_per_lifetime": target_reports,
        "current_reports_per_lifetime": rate * steps,
        "adult_steps_needed_at_current_rate": _share(target_reports, rate),
        "reports_per_adult_step_needed_at_current_lifespan": _share(target_reports, steps),
    }


def run(arm: str, seed: int, options: Any, grounded_fraction: float) -> OpportunityLedger:
    """Run one world and return its opportunity ledger."""
    point = harness.GridPoint(arm=arm, grounded_fraction=grounded_fraction, grounded_multiplier=1.0)
    adapter = harness.build_adapter(options.adapter_spec, seed, options.steps)
    world = harness.build_world(adapter, point, seed, options)
    ledger = OpportunityLedger()
    measurement_support.drive_world(harness, world, adapter, options.steps, ledger)
    return ledger


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = measurement_support.add_shared_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--arms", nargs="+", default=["ordinary", "oracle_invasion"])
    parser.add_argument("--target-reports", type=float, default=7.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure both reporting-opportunity throttles and print them per arm."""
    args = _parse_args(argv)
    options = measurement_support.harness_options(harness, args)
    print(f"steps={args.steps} seeds={list(args.seeds)} grounded={args.grounded_fraction}")
    for arm in args.arms:
        records = [
            record
            for seed in args.seeds
            for record in run(arm, seed, options, args.grounded_fraction).records
        ]
        lifespan = lifespan_metrics(records)
        funnel = funnel_metrics(records)
        print(f"\n=== arm={arm} ===")
        print("A. lifespan and mortality")
        for key, value in lifespan.items():
            print(f"   {key}: {value:.4f}")
        print("B. per-adult-step reporting funnel")
        for key, value in funnel.items():
            print(f"   {key}: {value:.4f}")
        print("C. silence and survival")
        for key, value in silence_survival_coupling(records).items():
            print(f"   {key}: {value:.4f}")
        print("D. what each throttle would have to deliver alone")
        for key, value in counterfactual_lifetime_reports(
            lifespan, funnel, args.target_reports
        ).items():
            print(f"   {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
