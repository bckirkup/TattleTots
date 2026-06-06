"""Cost accounting: surveillance, response, and damage cost tracking.

Accumulates per-step cost breakdowns from the domain adapter and
provides aggregate analytics for evaluating the information ecology's
operational efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepCosts:
    """Cost breakdown for a single simulation step."""

    time_step: int
    surveillance_cost: float = 0.0
    response_cost: float = 0.0
    damage_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.surveillance_cost + self.response_cost + self.damage_cost


@dataclass
class CostAccumulator:
    """Tracks cost history and provides aggregate analytics."""

    history: list[StepCosts] = field(default_factory=list)

    def record(self, costs: StepCosts) -> None:
        """Append a step cost record."""
        self.history.append(costs)

    def record_from_dict(self, time_step: int, cost_dict: dict[str, float]) -> None:
        """Record costs from a domain adapter's compute_costs() result."""
        self.history.append(
            StepCosts(
                time_step=time_step,
                surveillance_cost=cost_dict.get("surveillance_cost", 0.0),
                response_cost=cost_dict.get("response_cost", 0.0),
                damage_cost=cost_dict.get("damage_cost", 0.0),
            )
        )

    @property
    def total_surveillance(self) -> float:
        return sum(c.surveillance_cost for c in self.history)

    @property
    def total_response(self) -> float:
        return sum(c.response_cost for c in self.history)

    @property
    def total_damage(self) -> float:
        return sum(c.damage_cost for c in self.history)

    @property
    def total_cost(self) -> float:
        return self.total_surveillance + self.total_response + self.total_damage

    def cost_history(self) -> list[float]:
        """Total cost per step."""
        return [c.total for c in self.history]

    def surveillance_history(self) -> list[float]:
        """Surveillance cost per step."""
        return [c.surveillance_cost for c in self.history]

    def response_history(self) -> list[float]:
        """Response cost per step."""
        return [c.response_cost for c in self.history]

    def damage_history(self) -> list[float]:
        """Damage cost per step."""
        return [c.damage_cost for c in self.history]

    def mean_cost_per_step(self) -> float:
        """Average total cost per step."""
        if not self.history:
            return 0.0
        return self.total_cost / len(self.history)

    def summary(self) -> dict[str, float]:
        """Summary statistics for cost accounting."""
        return {
            "total_surveillance_cost": self.total_surveillance,
            "total_response_cost": self.total_response,
            "total_damage_cost": self.total_damage,
            "total_cost": self.total_cost,
            "mean_cost_per_step": self.mean_cost_per_step(),
            "steps_recorded": float(len(self.history)),
        }
