# How rich must a domain be for detector evolution to work?

The measurement chain in this repo produced four per-domain numbers
(`docs/cross-domain-grounding.md`) and three localized breaks
(`docs/currency-coupling-diagnosis.md`, `docs/heritability-measurement.md`,
`docs/reporting-opportunity-measurement.md`) but no *requirement* to compare a domain
against. This document supplies one analytically, so a candidate domain can be judged
before it is built, and answers the question it was written for: are agricultural
pests, wildfire initiation and fishery enforcement simply too thin to sustain this kind
of optimization, or is the shortfall somewhere else?

Short answer: event density is not the scarce dimension in any of the three. Two of the
three fail on a different dimension entirely — the best detector measured in them is
*worse than a constant guess* — and TattleTots itself fails on reward geometry, not on
richness. The scale needed to fix the remaining gap by population size alone is ~100 to
~550 breeding agents, not astronomical.

## Reproduction

```bash
uv run --no-sync --no-build python scripts/domain_richness_requirement.py
uv run --no-sync --no-build python scripts/domain_richness_requirement.py \
  --domains sparse_sensor --alignment 1.0 --population 120
```

The script is analytic: it runs no simulation, prints only, and every input is either a
measured number from a committed measurement doc or a stated assumption. Assumed inputs
are flagged in its output.

## The model

Selection cannot act on an agent's genotypic precision, only on its *observed*
correct-report rate, estimated from `k` verified reports. With between-genotype variance
`var_g`, within-genotype (environment) variance `var_env`, and pooled precision `p`:

| quantity | expression |
|---|---|
| reliability of an individual's observed precision | `rho(k) = var_g / (var_g + var_env + p(1-p)/k)` |
| response per generation | `R = b · i · sd_g · sqrt(rho(k))` |
| drift per generation (sd of the mean) | `D = sd_g / sqrt(N_e)` |

`i` is selection intensity (~0.8 when about half of adults reproduce), `N_e` the
effective breeding population, and `b` the alignment between relative fitness and
observed precision — the thing PR #58's payoff fixes moved from ~0 to +0.44. Requiring
`R >= z·D` gives a closed form for how many scored reports each agent needs:

```
rho* = z² / (b² i² N_e)
k*   = p(1-p) / ( var_g (1 - rho*) / rho* - var_env )
```

Two gates sit outside this arithmetic and cannot be traded against it:

1. **Exploitable margin** `M = p_ceiling - p_null`. If the best reachable precision does
   not beat the domain's own static-prior null, a constant guess dominates every
   detector and there is nothing for selection to climb, at any event density or
   population size.
2. **Reward geometry** `p_ceiling - p_break_even`, where
   `p_break_even = c_false_alarm / (c_false_alarm + v_correct)`. If break-even precision
   exceeds the reachable ceiling, silence is the optimum, `b` collapses to zero, and
   every term above is multiplied by nothing.

So domain richness is five-dimensional: exploitable margin, scored events per agent
lifetime `k`, effective population `N_e`, generations `G`, and reward geometry. Only the
middle three substitute for each other.

## What `k` is made of

`k` is not the domain's event rate. It is

```
k = adult_lifespan × P(evidence arrives) × P(anomaly clears threshold) × P(verified)
```

Measured on SparseSensor (`docs/reporting-opportunity-measurement.md`): lifespan 6.76
adult steps, evidence on **93%** of them, threshold cleared on **3.67%**, giving 0.457
reports per lifetime. The domain delivers roughly one usable observation per agent-step;
the engine's escalation threshold discards 96% of them. A domain cannot be blamed for a
factor that is lost downstream of it.

## Where each domain stands

`z = 2`, `i = 0.8`. SparseSensor's variances are measured (clone monocultures,
`var_g = 0.0145`, `var_env = 0.0085`); the three applied domains have not had clone
monocultures run, so they inherit those variances as an explicit assumption and their
`k` is unmeasured. Their margin and reward-geometry verdicts do not depend on either.

| domain | static-prior null | evolved precision | best reachable | exploitable margin | binding dimension |
|---|---:|---:|---:|---:|---|
| SparseSensor | 3.00% | 11.7% | 34.9% (best clone genome) | **+31.9 pp** | reward geometry |
| Coral Key (AIS/SAR) | 14.84% | 15.84% | 57.61% (designed reporter) | **+42.8 pp** | reward geometry |
| Scrapiron (fire, OPIR ablated) | 35.6% | 18.0% | 18.0% (no better arm measured) | **−17.6 pp** | exploitable margin |
| Xylella (grain, pests frozen) | 54.95% | 50.6% | 50.6% (no better arm measured) | **−4.4 pp** | exploitable margin |

Two distinct failures, not one:

- **Scrapiron and Xylella fail on margin.** Their instruments are so dominated by a
  well-placed constant guess (35.6% and 54.95% static priors) that no measured detector
  arm beats guessing. This is the one genuine *domain richness* failure in the set, and
  it is not about event density — both are event-dense. It is that the events are
  concentrated in a few predictable cells, so the *spatial/temporal predictability* of
  the hazard, not its frequency, sets the null. A domain whose hazards are dense but
  stereotyped is poor in exactly the dimension that matters. Their margins are also
  reported against ceilings measured only from evolved agents; a designed-reporter arm
  (as Coral has) could raise the ceiling and is the cheapest way to find out whether
  these two domains are irredeemable or merely unmeasured.
- **SparseSensor and Coral have ample margin and fail on price.** Break-even precision
  is 80% against a 34.9% ceiling, so reporting has negative expected return for every
  reachable genome — silence is the evolved optimum and `b ≈ 0`. Nothing about the
  domain is limiting here.

## The scale question, quantified

With the payoff knobs on (`b = 0.44`, measured in `docs/payoff-fix-measurement.md`) and
its variances, SparseSensor needs, to put the response 2 sd above drift:

| `b` | required `k` at `N_e = 60` | required `N_e` at the current `k = 0.457` |
|---:|---:|---:|
| 0.20 | not reachable at any `k` | 2,684 |
| 0.44 | 26.2 | 555 |
| 0.70 | 2.3 | 219 |
| 1.00 | 0.89 | 107 |

and the substitution between the two dimensions, at `z = 2`:

| `N_e` | required `k`, `b = 0.44` | required `k`, `b = 1.0` |
|---:|---:|---:|
| 60 | 26.16 | 0.89 |
| 120 | 3.34 | 0.40 |
| 250 | 1.16 | 0.19 |
| 500 | 0.51 | 0.09 |

Three things follow.

1. **The Foundation framing scales the wrong axis, but the right axis is cheap.**
   Statistical psychology needs many *individuals*; evolution needs many *scored events
   per individual* — and because `k*` falls as `1/N_e`, population size does substitute
   for sample size. The required scale is ~107 breeding agents at perfect fitness
   alignment and ~555 at the alignment actually measured, against a current cap of 60.
   That is one or two orders of magnitude short of "vast", and reachable on one CPU.
2. **Alignment is worth far more than scale.** Going from `b = 0.44` to `b = 1.0` cuts
   the required `k` by 29× and the required population by 5×. Repricing false alarms is
   a much better investment than a bigger world.
3. **The observation bar is not the selection bar.** The "7.2 reports per agent" figure
   in `docs/heritability-measurement.md` is what an *observer* needs to see heritability
   in a parent–child correlation (17.2 once within-genotype variance is included, which
   that figure omitted). Selection needs `rho*`, which at `b = 1` is 0.10 — i.e. `k* =
   0.89`, twice the current rate. We have been holding the engine to a measurement
   standard ~19× stricter than the evolutionary one.

## Verdict on the original question

Are the three domains too poor? Only in a specific and measurable sense, and not the
one suspected:

- Event density is adequate everywhere; evidence reaches agents on ~93% of adult steps.
- Fishery/IUU (Coral) is rich in the decisive dimension: +42.8 pp of exploitable margin,
  the largest in the set.
- Wildfire and grain are poor in the decisive dimension — hazard *predictability* makes
  a constant guess strong — and that is a domain-design property: they need either a
  harder-to-guess event process (more locations, less stereotyped timing) or a
  finer-grained decision to be scored on than "is there a hazard here".
- Meta-competence across hazard types cannot be selected for in any of them as built,
  because each publishes a single hazard type and instrument. That is a real limitation
  of the domain designs — but it is a breadth limitation, not a density or scale one.

## What this changes, and what it does not

This document adds an analytic script, tests, and no engine behavior: no defaults move,
no scaffolding is added, and neither falsification clause is claimed. It reorders the
remaining levers by expected value:

1. reprice false alarms against the instrument's reachable precision (raises `b`, the
   highest-leverage term, and the cause of the current binding failure);
2. calibrate the escalation threshold to the agent's own anomaly distribution (worth up
   to ~21× in `k` at no cost in population: the oracle policy escalates on 78% of adult
   steps in the same worlds against the evolved arm's 3.67%);
3. run designed-reporter arms in Scrapiron and Xylella to establish whether their
   ceilings clear their nulls at all;
4. only then consider raising `N_e` — and if so, to ~250, not to Trantor.
