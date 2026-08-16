"""Per-agent payoff ledger: does report correctness pay in either currency?

The ecology only evolves competence if correctness changes an agent's currency
balances and, through them, its offspring count. This ledger observes a running
`World` and records, per agent, the correctness it achieved alongside every
currency inflow it received, so the correctness -> income -> reproduction chain
can be measured link by link instead of inferred from population-level rates.

Nothing here is domain-specific: it reads public agent state and user trust only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tattletots.models.agent import Agent, LifecycleStage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tattletots.engine.world import World


@dataclass
class AgentPayoffRecord:
    """Lifetime correctness and currency inflows for a single agent."""

    agent_id: str
    generation: int = 0
    parent_ids: tuple[str, ...] = ()
    steps_alive: int = 0
    adult_steps: int = 0
    reports_issued: int = 0
    correct_reports: int = 0
    false_alarms: int = 0
    attention_income: float = 0.0
    information_yield: float = 0.0
    information_subsidy: float = 0.0
    trust_samples: float = 0.0
    trust_observations: int = 0
    false_alarm_price_sum: float = 0.0
    correct_report_value_sum: float = 0.0
    final_information_energy: float = 0.0
    final_attention_energy: float = 0.0
    offspring: int = 0

    @property
    def mean_user_trust(self) -> float:
        """Mean trust held in this agent across observed steps and users."""
        if self.trust_observations == 0:
            return 0.0
        return self.trust_samples / self.trust_observations

    @property
    def attention_income_per_step(self) -> float:
        """Attention inflow per step alive, comparable across lifespans."""
        if self.steps_alive == 0:
            return 0.0
        return self.attention_income / self.steps_alive

    @property
    def mean_false_alarm_price(self) -> float:
        """Attention charged per false alarm, averaged over the steps this agent lived."""
        if self.steps_alive == 0:
            return 0.0
        return self.false_alarm_price_sum / self.steps_alive

    @property
    def mean_correct_report_value(self) -> float:
        """Attention a correct report would have earned, averaged over steps alive."""
        if self.steps_alive == 0:
            return 0.0
        return self.correct_report_value_sum / self.steps_alive

    @property
    def realized_break_even_precision(self) -> float:
        """Precision at which this agent's reporting return changes sign.

        `price / (price + value)`: the precision that makes `p*value - (1-p)*price`
        zero. It is what the agent actually faced, whether the price came from the flat
        penalty or from repricing.
        """
        price = self.mean_false_alarm_price
        value = self.mean_correct_report_value
        if price + value <= 0.0:
            return 0.0
        return price / (price + value)

    @property
    def information_income_per_step(self) -> float:
        """Information inflow (yield plus peer subsidy) per step alive."""
        if self.steps_alive == 0:
            return 0.0
        return (self.information_yield + self.information_subsidy) / self.steps_alive

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat mapping for serialization."""
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "n_parents": len(self.parent_ids),
            "steps_alive": self.steps_alive,
            "adult_steps": self.adult_steps,
            "reports_issued": self.reports_issued,
            "correct_reports": self.correct_reports,
            "false_alarms": self.false_alarms,
            "attention_income": self.attention_income,
            "information_yield": self.information_yield,
            "information_subsidy": self.information_subsidy,
            "mean_user_trust": self.mean_user_trust,
            "attention_income_per_step": self.attention_income_per_step,
            "information_income_per_step": self.information_income_per_step,
            "final_information_energy": self.final_information_energy,
            "final_attention_energy": self.final_attention_energy,
            "mean_false_alarm_price": self.mean_false_alarm_price,
            "mean_correct_report_value": self.mean_correct_report_value,
            "realized_break_even_precision": self.realized_break_even_precision,
            "offspring": self.offspring,
        }


@dataclass
class ReproductionGate:
    """How reproduction opportunities were rationed over the run."""

    eligible_agent_steps: int = 0
    agent_steps: int = 0
    co_limited_agent_steps: int = 0
    attention_limited_agent_steps: int = 0
    information_limited_agent_steps: int = 0
    population_capped_steps: int = 0
    steps: int = 0

    def as_dict(self) -> dict[str, float]:
        """Shares of agent-steps in each reproduction-gating condition."""
        agent_steps = max(self.agent_steps, 1)
        return {
            "eligible_share": self.eligible_agent_steps / agent_steps,
            "co_limited_share": self.co_limited_agent_steps / agent_steps,
            "attention_limited_share": self.attention_limited_agent_steps / agent_steps,
            "information_limited_share": self.information_limited_agent_steps / agent_steps,
            "population_capped_step_share": self.population_capped_steps / max(self.steps, 1),
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, returning 0.0 for degenerate inputs."""
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if math.isclose(float(x.std()), 0.0, abs_tol=1e-12) or math.isclose(
        float(y.std()), 0.0, abs_tol=1e-12
    ):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _correlate_pairs(pairs: list[tuple[float, float]]) -> float:
    """Pearson correlation over (parent, child) trait pairs."""
    return _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def _precision(record: AgentPayoffRecord) -> float:
    """Share of an agent's reports that were verified correct."""
    if record.reports_issued == 0:
        return 0.0
    return record.correct_reports / record.reports_issued


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys on xs, returning 0.0 for degenerate inputs."""
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if math.isclose(float(x.std()), 0.0, abs_tol=1e-12):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


class PayoffLedger:
    """Accumulate per-agent correctness and currency inflows across a run."""

    def __init__(self) -> None:
        self._records: dict[str, AgentPayoffRecord] = {}
        self.gate = ReproductionGate()
        self._trust_break_even_precision = 0.0
        self._false_alarm_penalty = 0.0

    def _record_for(self, agent: Agent) -> AgentPayoffRecord:
        record = self._records.get(agent.id)
        if record is None:
            record = AgentPayoffRecord(
                agent_id=agent.id,
                generation=agent.state.generation,
            )
            self._records[agent.id] = record
        return record

    def observe(self, world: World) -> None:
        """Accumulate one completed step of the supplied world."""
        living = [agent for agent in world.agents.values() if agent.is_alive]
        users = list(world.users.values())
        config = world.config
        trust_span = config.trust_delta_pos + config.trust_delta_neg
        self._trust_break_even_precision = (
            config.trust_delta_neg / trust_span if trust_span else 0.0
        )
        self._false_alarm_penalty = config.false_alarm_penalty
        self.gate.steps += 1
        if len(living) >= world.config.max_population:
            self.gate.population_capped_steps += 1

        for agent in living:
            record = self._record_for(agent)
            record.steps_alive += 1
            if agent.state.lifecycle == LifecycleStage.ADULT:
                record.adult_steps += 1
            record.attention_income += agent.state.last_step_attention_income
            record.information_yield += agent.state.last_step_yield
            record.information_subsidy += agent.state.last_step_info_subsidy
            record.false_alarm_price_sum += agent.state.last_step_false_alarm_price
            record.correct_report_value_sum += agent.state.last_step_correct_report_value
            for user in users:
                record.trust_samples += user.get_trust(agent.id)
                record.trust_observations += 1
            self._observe_gate(agent, world)

    def _observe_gate(self, agent: Agent, world: World) -> None:
        config = world.config
        self.gate.agent_steps += 1
        if agent.can_reproduce:
            self.gate.eligible_agent_steps += 1
        factor = agent.reproduction_limiting_factor(
            config.reproduction_coupling_strength,
            config.reproduction_information_scale,
            config.reproduction_attention_scale,
        )
        if factor < 1.0:
            self.gate.co_limited_agent_steps += 1
            threshold = agent.genome.reproduction_threshold
            info_required = (
                threshold * agent.genome.information_requirement
            ) * config.reproduction_information_scale
            attn_required = (
                threshold * agent.genome.attention_requirement
            ) * config.reproduction_attention_scale
            info_sufficiency = (
                agent.state.energy.information / info_required if info_required else 1.0
            )
            attn_sufficiency = (
                agent.state.energy.attention / attn_required if attn_required else 1.0
            )
            if attn_sufficiency < info_sufficiency:
                self.gate.attention_limited_agent_steps += 1
            else:
                self.gate.information_limited_agent_steps += 1

    def finalize(self, world: World) -> None:
        """Attach terminal reserves, report counts, and offspring counts."""
        for agent in world.agents.values():
            record = self._record_for(agent)
            record.generation = agent.state.generation
            record.parent_ids = tuple(agent.state.parent_ids)
            record.reports_issued = agent.state.reports_issued
            record.correct_reports = agent.state.correct_reports
            record.false_alarms = agent.state.false_alarms
            record.final_information_energy = agent.state.energy.information
            record.final_attention_energy = agent.state.energy.attention
        for agent in world.agents.values():
            for parent_id in agent.state.parent_ids:
                parent = self._records.get(parent_id)
                if parent is not None:
                    parent.offspring += 1

    @property
    def records(self) -> list[AgentPayoffRecord]:
        """All observed agents, including those that died during the run."""
        return list(self._records.values())

    def coupling_summary(self) -> dict[str, Any]:
        """Link-by-link coupling from correctness to currency to offspring."""
        adults = [record for record in self.records if record.adult_steps > 0]
        if not adults:
            return {"n_adults": 0}

        correct = [float(record.correct_reports) for record in adults]
        precision = [
            record.correct_reports / record.reports_issued if record.reports_issued else 0.0
            for record in adults
        ]
        trust = [record.mean_user_trust for record in adults]
        attn = [record.attention_income_per_step for record in adults]
        info = [record.information_income_per_step for record in adults]
        offspring = [float(record.offspring) for record in adults]

        reports = [float(record.reports_issued) for record in adults]
        correct_group = [record for record in adults if record.correct_reports > 0]
        incorrect_group = [
            record for record in adults if record.correct_reports == 0 and record.reports_issued > 0
        ]
        silent_group = [record for record in adults if record.reports_issued == 0]
        reporting_group = [record for record in adults if record.reports_issued > 0]

        return {
            "n_adults": len(adults),
            "n_with_correct_report": len(correct_group),
            "n_reporting_never_correct": len(incorrect_group),
            "corr_correct_reports_trust": _pearson(correct, trust),
            "corr_precision_trust": _pearson(precision, trust),
            "corr_trust_attention_income": _pearson(trust, attn),
            "corr_correct_reports_attention_income": _pearson(correct, attn),
            "corr_correct_reports_information_income": _pearson(correct, info),
            "corr_attention_income_offspring": _pearson(attn, offspring),
            "corr_information_income_offspring": _pearson(info, offspring),
            "corr_correct_reports_offspring": _pearson(correct, offspring),
            "corr_reports_issued_attention_income": _pearson(reports, attn),
            "corr_reports_issued_offspring": _pearson(reports, offspring),
            "n_silent_adults": len(silent_group),
            "silent_adult_share": len(silent_group) / len(adults),
            "mean_reports_per_adult": _mean(reports),
            "mean_adult_steps": _mean([float(record.adult_steps) for record in adults]),
            "silent_mean_attention_income": _mean(
                [record.attention_income_per_step for record in silent_group]
            ),
            "reporting_mean_attention_income": _mean(
                [record.attention_income_per_step for record in reporting_group]
            ),
            "silent_mean_offspring": _mean([float(record.offspring) for record in silent_group]),
            "reporting_mean_offspring": _mean(
                [float(record.offspring) for record in reporting_group]
            ),
            "trust_break_even_precision": self._trust_break_even_precision,
            "false_alarm_penalty_in_attention_income_steps": (
                self._false_alarm_penalty / _mean(attn) if _mean(attn) > 0.0 else 0.0
            ),
            "mean_false_alarm_price": _mean([record.mean_false_alarm_price for record in adults]),
            "mean_correct_report_value": _mean(
                [record.mean_correct_report_value for record in adults]
            ),
            "realized_break_even_precision": _mean(
                [
                    record.realized_break_even_precision
                    for record in adults
                    if record.mean_false_alarm_price + record.mean_correct_report_value > 0.0
                ]
            ),
            "mean_attention_income_per_step": _mean(attn),
            "mean_information_income_per_step": _mean(info),
            "mean_information_share_of_reserves": _mean(
                [
                    record.final_information_energy
                    / (record.final_information_energy + record.final_attention_energy)
                    for record in adults
                    if (record.final_information_energy + record.final_attention_energy) > 0.0
                ]
            ),
            "mean_subsidy_share_of_information_income": _mean(
                [
                    record.information_subsidy
                    / (record.information_yield + record.information_subsidy)
                    for record in adults
                    if (record.information_yield + record.information_subsidy) > 0.0
                ]
            ),
            "correct_group_mean_offspring": _mean(
                [float(record.offspring) for record in correct_group]
            ),
            "never_correct_group_mean_offspring": _mean(
                [float(record.offspring) for record in incorrect_group]
            ),
            "correct_group_mean_attention_income": _mean(
                [record.attention_income_per_step for record in correct_group]
            ),
            "never_correct_group_mean_attention_income": _mean(
                [record.attention_income_per_step for record in incorrect_group]
            ),
            "reproduction_gate": self.gate.as_dict(),
            **self._falsification_summary(adults),
        }

    def _falsification_summary(self, adults: list[AgentPayoffRecord]) -> dict[str, Any]:
        """The two falsification clauses, measured on this run.

        Clause 1 is a within-run rise in correct-report rate over generations at
        fixed initial parameters; clause 2 is parent-child reproductive correlation.
        """
        by_generation: dict[int, list[AgentPayoffRecord]] = {}
        for record in adults:
            by_generation.setdefault(record.generation, []).append(record)
        precision_by_generation = {
            generation: (
                sum(record.correct_reports for record in cohort)
                / sum(record.reports_issued for record in cohort)
                if sum(record.reports_issued for record in cohort)
                else 0.0
            )
            for generation, cohort in sorted(by_generation.items())
        }
        reporting_generations = [
            generation
            for generation, cohort in sorted(by_generation.items())
            if sum(record.reports_issued for record in cohort)
        ]
        return {
            "precision_by_generation": precision_by_generation,
            "generations_observed": len(reporting_generations),
            "precision_generation_slope": _slope(
                [float(generation) for generation in reporting_generations],
                [precision_by_generation[generation] for generation in reporting_generations],
            ),
            "corr_parent_child_offspring": _correlate_pairs(
                self._parent_child_pairs(lambda record: float(record.offspring))
            ),
            "n_parent_child_pairs": len(
                self._parent_child_pairs(lambda record: float(record.offspring))
            ),
            "corr_parent_child_precision": _correlate_pairs(
                self._parent_child_pairs(_precision, reporting_only=True)
            ),
            "n_parent_child_precision_pairs": len(
                self._parent_child_pairs(_precision, reporting_only=True)
            ),
        }

    def _parent_child_pairs(
        self,
        trait: Callable[[AgentPayoffRecord], float],
        reporting_only: bool = False,
    ) -> list[tuple[float, float]]:
        """Parent/child values of one trait over every observed parentage link."""
        pairs: list[tuple[float, float]] = []
        for record in self.records:
            if reporting_only and record.reports_issued == 0:
                continue
            for parent_id in record.parent_ids:
                parent = self._records.get(parent_id)
                if parent is None or (reporting_only and parent.reports_issued == 0):
                    continue
                pairs.append((trait(parent), trait(record)))
        return pairs


__all__ = [
    "AgentPayoffRecord",
    "PayoffLedger",
    "ReproductionGate",
]
