"""Dual-currency energy accounting for agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnergyReserves(BaseModel):
    """Dual energy reserves tracking both information and attention currencies.

    An agent dies if either reserve hits zero.
    - Information energy: earned by compressing data, received as downstream subsidy.
    - Attention energy: earned by having reports valued by human users.
    """

    information: float = Field(default=1.0, description="Information energy reserve")
    attention: float = Field(default=1.0, description="Attention energy reserve")

    @property
    def is_alive(self) -> bool:
        """Agent persists only if BOTH reserves are positive."""
        return self.information > 0.0 and self.attention > 0.0

    @property
    def total(self) -> float:
        """Combined energy (used for reproduction threshold checks)."""
        return self.information + self.attention

    def apply_info_delta(self, delta: float) -> None:
        """Apply information energy change (yield - cost)."""
        self.information += delta

    def apply_attention_delta(self, delta: float) -> None:
        """Apply attention energy change (income - maintenance - penalties)."""
        self.attention += delta
