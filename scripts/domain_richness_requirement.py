#!/usr/bin/env python3
"""How rich must a domain be, in which dimensions, for detector evolution to work?

The measurement chain in `docs/cross-domain-grounding.md`,
`docs/heritability-measurement.md` and `docs/reporting-opportunity-measurement.md`
produced four numbers per domain but no requirement to compare them against. This
script supplies the requirement analytically, so a domain can be judged before it is
simulated.

The model is the breeder's equation with a noisy indicator trait. Selection can only
act on an agent's *observed* correct-report rate, estimated from `k` verified reports:

    reliability      rho(k) = var_g / (var_g + var_env + p(1-p)/k)
    response/gen     R      = b * i * sd_g * sqrt(rho(k))
    drift/gen (sd)   D      = sd_g / sqrt(N_e)

where `var_g` is between-genotype variance in precision, `var_env` the within-genotype
environmental variance, `i` the selection intensity, `b` the alignment between relative
fitness and observed precision, and `N_e` the effective breeding population. Requiring
the response to clear drift by `z` standard deviations gives a closed form for the
reports each agent must be scored on:

    rho* = z^2 / (b^2 * i^2 * N_e)
    k*   = p(1-p) / (var_g * (1 - rho*) / rho* - var_env)

k* falls as N_e rises: population size and per-agent sample size are substitutes, and
the script reports the exchange rate between them. Two further gates do not trade off
against anything:

  * exploitable margin -- the best reachable precision must beat the domain's own
    static-prior null, else a constant guess dominates any detector and there is
    nothing to select for however dense the events are; and
  * reward geometry -- the break-even precision implied by the false-alarm penalty and
    the attention income must sit below that reachable precision, else silence is the
    optimum, `b` collapses to zero, and every other dimension is moot.

Prints only; it writes no artifacts.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

MEASUREMENT_RELIABILITY = 0.5
"""Reliability an *observer* needs to see heritability, not the reliability selection
needs to act: the 7.2 reports/agent figure in `docs/heritability-measurement.md` is
this bar, and it is far stricter than `rho*`."""


@dataclass(frozen=True)
class DomainProfile:
    """The measured richness of one domain, in the dimensions the model needs.

    `reports_per_lifetime` and `ceiling_source` are the only fields that may be
    unmeasured; everything else is either measured or an explicit stated assumption.
    """

    name: str
    static_prior_null: float
    mean_precision: float
    ceiling_precision: float
    genetic_variance: float
    environment_variance: float
    effective_population: float
    generations: float
    selection_intensity: float
    fitness_alignment: float
    break_even_precision: float
    reports_per_lifetime: float | None = None
    assumed_variances: bool = False
    source: str = ""


def reliability(profile: DomainProfile, reports: float) -> float:
    """Fraction of observed-precision variance that is genotypic at `reports` reports."""
    if reports <= 0.0:
        return 0.0
    p = profile.mean_precision
    sampling = p * (1.0 - p) / reports
    total = profile.genetic_variance + profile.environment_variance + sampling
    if total <= 0.0:
        return 0.0
    return profile.genetic_variance / total


def response_per_generation(profile: DomainProfile, reports: float) -> float:
    """Expected per-generation gain in mean genotypic precision."""
    sd_g = math.sqrt(profile.genetic_variance)
    return (
        profile.fitness_alignment
        * profile.selection_intensity
        * sd_g
        * math.sqrt(reliability(profile, reports))
    )


def drift_sd_per_generation(profile: DomainProfile) -> float:
    """Standard deviation of the drift-only change in mean genotypic precision."""
    if profile.effective_population <= 0.0:
        return math.inf
    return math.sqrt(profile.genetic_variance / profile.effective_population)


def required_reliability(profile: DomainProfile, z: float) -> float:
    """Reliability at which the selection response clears drift by `z` sigma."""
    denominator = (
        profile.fitness_alignment**2 * profile.selection_intensity**2 * profile.effective_population
    )
    if denominator <= 0.0:
        return math.inf
    return z * z / denominator


def required_reports(profile: DomainProfile, z: float) -> float:
    """Verified reports per agent needed to reach `required_reliability`.

    Returns `inf` when no sample size suffices, i.e. when the genotypic variance is
    too small against the within-genotype environmental variance -- in that case the
    domain needs repeated environments per genotype, not more reports.
    """
    target = required_reliability(profile, z)
    if not math.isfinite(target) or target >= 1.0:
        return math.inf
    headroom = profile.genetic_variance * (1.0 - target) / target - profile.environment_variance
    if headroom <= 0.0:
        return math.inf
    p = profile.mean_precision
    return p * (1.0 - p) / headroom


def reports_for_reliability(profile: DomainProfile, target: float) -> float:
    """Reports needed to reach an arbitrary reliability target (e.g. an observer's)."""
    if not 0.0 < target < 1.0:
        return math.inf
    headroom = profile.genetic_variance * (1.0 - target) / target - profile.environment_variance
    if headroom <= 0.0:
        return math.inf
    p = profile.mean_precision
    return p * (1.0 - p) / headroom


def required_population(profile: DomainProfile, z: float) -> float:
    """Effective population that makes the domain's *current* report count sufficient.

    This is the substitution the Foundation framing appeals to: with too few scored
    events per agent, more agents can still deliver a response above drift.
    """
    reports = profile.reports_per_lifetime
    if reports is None or reports <= 0.0:
        return math.inf
    rho = reliability(profile, reports)
    denominator = profile.fitness_alignment**2 * profile.selection_intensity**2 * rho
    if denominator <= 0.0:
        return math.inf
    return z * z / denominator


def exploitable_margin(profile: DomainProfile) -> float:
    """Best reachable precision minus the null a constant guess already achieves."""
    return profile.ceiling_precision - profile.static_prior_null


def generations_to_ceiling(profile: DomainProfile, reports: float) -> float:
    """Generations for the response to consume the headroom below the ceiling."""
    response = response_per_generation(profile, reports)
    if response <= 0.0:
        return math.inf
    headroom = profile.ceiling_precision - profile.mean_precision
    if headroom <= 0.0:
        return 0.0
    return headroom / response


def evaluate(profile: DomainProfile, z: float) -> dict[str, float]:
    """Score one domain against every dimension of the requirement."""
    reports = profile.reports_per_lifetime
    metrics: dict[str, float] = {
        "exploitable_margin": exploitable_margin(profile),
        "ceiling_minus_break_even": profile.ceiling_precision - profile.break_even_precision,
        "required_reliability": required_reliability(profile, z),
        "required_reports_for_selection": required_reports(profile, z),
        "required_reports_for_observation": reports_for_reliability(
            profile, MEASUREMENT_RELIABILITY
        ),
        "drift_sd_per_generation": drift_sd_per_generation(profile),
        "generations_available": profile.generations,
    }
    if reports is not None:
        metrics["reports_per_lifetime"] = reports
        metrics["reliability_at_current_reports"] = reliability(profile, reports)
        metrics["response_per_generation"] = response_per_generation(profile, reports)
        metrics["response_over_drift"] = (
            metrics["response_per_generation"] / metrics["drift_sd_per_generation"]
            if metrics["drift_sd_per_generation"] > 0.0
            else math.inf
        )
        metrics["generations_to_ceiling"] = generations_to_ceiling(profile, reports)
        metrics["required_population_at_current_reports"] = required_population(profile, z)
        needed = metrics["required_reports_for_selection"]
        metrics["report_shortfall_factor"] = (
            needed / reports if reports > 0.0 and math.isfinite(needed) else math.inf
        )
    return metrics


def binding_dimension(profile: DomainProfile, z: float) -> str:
    """Name the first dimension that fails, in the order the model makes them matter."""
    metrics = evaluate(profile, z)
    if metrics["exploitable_margin"] <= 0.0:
        return "exploitable margin (a constant guess beats the best detector)"
    if metrics["ceiling_minus_break_even"] <= 0.0:
        return "reward geometry (silence pays better than any reachable precision)"
    if profile.fitness_alignment <= 0.0:
        return "fitness alignment (payoff does not track observed precision)"
    reports = profile.reports_per_lifetime
    if reports is None:
        return "unmeasured reports per lifetime"
    if metrics["response_over_drift"] < z:
        return "scored events per agent (response below the drift bar)"
    if profile.generations < 1.0:
        return "generations available"
    return "none: the domain supplies enough for a response above drift"


# Measured profiles. Sources are the committed measurement docs in this repo and the
# per-domain PRs they cite; every assumption is flagged rather than folded in silently.
_SPARSE_SENSOR = DomainProfile(
    name="sparse_sensor",
    static_prior_null=0.0300,
    mean_precision=0.117,
    ceiling_precision=0.349,
    genetic_variance=0.0145,
    environment_variance=0.0085,
    effective_population=60.0,
    generations=15.4,
    selection_intensity=0.8,
    fitness_alignment=0.0,
    break_even_precision=0.80,
    reports_per_lifetime=0.457,
    source="docs/heritability-measurement.md, docs/currency-coupling-diagnosis.md",
)

_PROFILES: dict[str, DomainProfile] = {
    "sparse_sensor": _SPARSE_SENSOR,
    "sparse_sensor_payoff_knobs": replace(
        _SPARSE_SENSOR,
        name="sparse_sensor_payoff_knobs",
        fitness_alignment=0.44,
        source="docs/payoff-fix-measurement.md (correctness->offspring +0.44)",
    ),
    "coral": replace(
        _SPARSE_SENSOR,
        name="coral",
        static_prior_null=0.1484,
        mean_precision=0.1584,
        ceiling_precision=0.5761,
        generations=15.4,
        fitness_alignment=0.0,
        reports_per_lifetime=None,
        assumed_variances=True,
        source="docs/cross-domain-grounding.md (Coral PR #29); variances assumed",
    ),
    "scrapiron": replace(
        _SPARSE_SENSOR,
        name="scrapiron",
        static_prior_null=0.356,
        mean_precision=0.180,
        ceiling_precision=0.180,
        fitness_alignment=0.0,
        reports_per_lifetime=None,
        assumed_variances=True,
        source="docs/cross-domain-grounding.md (Scrapiron PR #27); variances assumed",
    ),
    "xylella": replace(
        _SPARSE_SENSOR,
        name="xylella",
        static_prior_null=0.5495,
        mean_precision=0.506,
        ceiling_precision=0.506,
        generations=30.8,
        fitness_alignment=0.0,
        reports_per_lifetime=None,
        assumed_variances=True,
        source="docs/cross-domain-grounding.md (Xylella PR #29); variances assumed",
    ),
}


def profiles() -> dict[str, DomainProfile]:
    """Return the committed domain profiles keyed by name."""
    return dict(_PROFILES)


def _print_profile(profile: DomainProfile, z: float) -> None:
    print(f"\n=== {profile.name} ===")
    if profile.source:
        print(f"  source: {profile.source}")
    if profile.assumed_variances:
        print("  NOTE: genetic/environmental variances assumed from sparse_sensor")
    metrics = evaluate(profile, z)
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"  binding dimension: {binding_dimension(profile, z)}")


def _print_report_sweep(profile: DomainProfile, reports_grid: list[float]) -> None:
    print(f"\n  reliability and response vs reports per agent ({profile.name}):")
    for reports in reports_grid:
        rho = reliability(profile, reports)
        response = response_per_generation(profile, reports)
        print(f"    k={reports:>7.2f}  rho={rho:.4f}  response/gen={response:.5f}")


def _with_overrides(profile: DomainProfile, args: argparse.Namespace) -> DomainProfile:
    """Apply the CLI exchange-rate overrides, so the substitutions can be explored."""
    changes: dict[str, float] = {}
    if args.alignment is not None:
        changes["fitness_alignment"] = args.alignment
    if args.population is not None:
        changes["effective_population"] = args.population
    if args.reports is not None:
        changes["reports_per_lifetime"] = args.reports
    return replace(profile, **changes) if changes else profile


def main() -> None:
    """Print the requirement and each domain's standing against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        help="profile names, or 'all'",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=2.0,
        help="how many drift standard deviations the response must clear",
    )
    parser.add_argument(
        "--report-grid",
        type=float,
        nargs="+",
        default=[0.457, 1.0, 2.0, 5.0, 7.2, 20.0],
        help="reports-per-agent values for the sensitivity sweep",
    )
    parser.add_argument(
        "--alignment",
        type=float,
        default=None,
        help="override corr(relative fitness, observed precision) for every profile",
    )
    parser.add_argument(
        "--population",
        type=float,
        default=None,
        help="override the effective breeding population for every profile",
    )
    parser.add_argument(
        "--reports",
        type=float,
        default=None,
        help="override measured reports per agent lifetime for every profile",
    )
    args = parser.parse_args()

    available = profiles()
    names = list(available) if "all" in args.domains else list(args.domains)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown domains: {unknown}; available: {list(available)}")

    print(f"drift bar z = {args.z}")
    for name in names:
        profile = _with_overrides(available[name], args)
        _print_profile(profile, args.z)
        _print_report_sweep(profile, list(args.report_grid))


if __name__ == "__main__":
    main()
