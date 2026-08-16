"""Tests for the analytic domain-richness requirement.

The model is closed-form, so the tests assert on the *shape* of the response to each
dimension (ordering, live knobs, substitution) plus the bounds every quantity must
respect, rather than on point values.
"""

from __future__ import annotations

import math

import pytest

from script_loading import load_script

script = load_script("domain_richness_requirement")


def _profile(**overrides: object) -> object:
    base = script.profiles()["sparse_sensor"]
    return script.replace(base, **overrides) if overrides else base


def test_reliability_grades_with_reports_and_stays_bounded() -> None:
    profile = _profile()
    grid = [0.25, 1.0, 4.0, 16.0, 64.0]
    values = [script.reliability(profile, k) for k in grid]

    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[-1] - values[0] > 0.2, f"reliability looks dead: {values}"
    assert script.reliability(profile, 0.0) == pytest.approx(0.0)


def test_reliability_saturates_at_the_environmental_ceiling() -> None:
    """With infinite reports the estimate is still limited by within-genotype variance."""
    profile = _profile()
    ceiling = profile.genetic_variance / (profile.genetic_variance + profile.environment_variance)

    assert script.reliability(profile, 1e9) == pytest.approx(ceiling, rel=1e-4)
    assert script.reliability(profile, 4.0) < ceiling


def test_response_grades_with_fitness_alignment_and_vanishes_without_it() -> None:
    values = [
        script.response_per_generation(_profile(fitness_alignment=b), 5.0)
        for b in (0.0, 0.25, 0.5, 1.0)
    ]

    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] > 0.02, f"alignment looks dead: {values}"


def test_required_reports_falls_as_population_rises() -> None:
    """Population size and per-agent sample size are substitutes."""
    values = [
        script.required_reports(_profile(fitness_alignment=1.0, effective_population=n), 2.0)
        for n in (60.0, 120.0, 250.0, 500.0)
    ]

    assert values == sorted(values, reverse=True)
    assert all(value > 0.0 for value in values)
    assert values[0] / values[-1] > 4.0, f"substitution looks dead: {values}"


def test_required_population_falls_as_alignment_rises() -> None:
    values = [
        script.required_population(_profile(fitness_alignment=b), 2.0)
        for b in (0.2, 0.44, 0.7, 1.0)
    ]

    assert values == sorted(values, reverse=True)
    assert values[0] / values[-1] > 4.0


def test_no_sample_size_suffices_when_environmental_noise_dominates() -> None:
    """A domain whose genotype effect is swamped per environment needs replicates."""
    starved = _profile(
        fitness_alignment=1.0,
        genetic_variance=0.0005,
        environment_variance=0.05,
    )

    assert not math.isfinite(script.required_reports(starved, 2.0))
    assert not math.isfinite(script.required_reports(_profile(fitness_alignment=0.0), 2.0))


def test_generations_to_ceiling_falls_as_response_rises() -> None:
    values = [
        script.generations_to_ceiling(_profile(fitness_alignment=b), 5.0) for b in (0.1, 0.3, 1.0)
    ]

    assert values == sorted(values, reverse=True)
    assert math.isfinite(values[-1])
    assert not math.isfinite(script.generations_to_ceiling(_profile(fitness_alignment=0.0), 5.0))


def test_measured_domains_split_on_exploitable_margin() -> None:
    """Two of the three real domains price a constant guess above the best detector."""
    margins = {
        name: script.exploitable_margin(profile) for name, profile in script.profiles().items()
    }

    assert margins["sparse_sensor"] > 0.0
    assert margins["coral"] > 0.0
    assert margins["scrapiron"] < 0.0
    assert margins["xylella"] < 0.0


def test_binding_dimension_reports_margin_before_reward_geometry() -> None:
    both_broken = _profile(
        ceiling_precision=0.10,
        static_prior_null=0.30,
        break_even_precision=0.80,
    )
    reward_only = _profile(ceiling_precision=0.30, break_even_precision=0.80)
    aligned = _profile(
        fitness_alignment=1.0,
        break_even_precision=0.05,
        effective_population=500.0,
        reports_per_lifetime=5.0,
    )

    assert "exploitable margin" in script.binding_dimension(both_broken, 2.0)
    assert "reward geometry" in script.binding_dimension(reward_only, 2.0)
    assert script.binding_dimension(aligned, 2.0).startswith("none")


def test_metrics_are_finite_or_infinite_never_nan() -> None:
    for profile in script.profiles().values():
        for key, value in script.evaluate(profile, 2.0).items():
            assert not math.isnan(value), key


def test_generation_budget_does_not_move_the_per_generation_response() -> None:
    """Negative control: a dimension the response does not use must not move it."""
    base = script.response_per_generation(_profile(fitness_alignment=0.5), 5.0)
    other = script.response_per_generation(_profile(fitness_alignment=0.5, generations=1000.0), 5.0)

    assert other == pytest.approx(base, rel=1e-12)


def test_observation_bar_is_stricter_than_the_selection_bar() -> None:
    """Seeing heritability needs more reports than acting on it does."""
    profile = _profile(fitness_alignment=1.0)
    selection = script.required_reports(profile, 2.0)
    observation = script.reports_for_reliability(profile, script.MEASUREMENT_RELIABILITY)

    assert observation > selection
