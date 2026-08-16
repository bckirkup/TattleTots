# Lever 2: calibrating escalation thresholds in the units they are compared in

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/measure_threshold_calibration.py
uv run --no-sync --no-build python scripts/check_falsification_reliability.py \
  --doses 8.0 --break-even-precision 0.2 --score-units --threshold-range 0.05 0.3
```

Generated artifacts: `docs/threshold-calibration.md`, `docs/threshold-calibration.json`.
Instrument: `SparseSensorScenario` (published coordinates, graded latent source, 3.00%
static-prior localization null). Lever 1 is held fixed in every arm
(`correct_report_attention_value=8`, `false_alarm_break_even_precision=0.2`,
merit-ordered reproduction on), so the only differences between arms are the calibration
units and the starting range of the `escalation_threshold` trait.

## The unit mismatch

`escalation.should_escalate` compares a *normalized* anomaly score against the effective
threshold. `normalize_anomaly` maps the raw anomaly through a rolling z-score and a
logistic, so the score it returns is always in `(0, 1)`. But the adaptive threshold modes
in `compute_effective_threshold` took their quantile or volatility band from
`agent.state.anomaly_history`, which stores *raw* anomaly values whose scale is set by
the agent's compression model. The two sides of the comparison were in different units.

A one-seed instrumented run (200 steps, ordinary arm, lever 1 on) shows what that does to
the decision:

| Mode | Decisions | Raw median | Raw p90 | Normalized median | Threshold median | Fired |
|---|---:|---:|---:|---:|---:|---:|
| `adaptive_quantile` | 965 | 0.013 | 1.635 | 0.238 | 0.032 | 74.4% |
| `adaptive_volatility` | 265 | 0.002 | 0.021 | 0.242 | 0.524 | 11.3% |
| `fixed` | 1214 | 0.015 | 2.352 | 0.265 | 0.540 | 10.8% |

The adaptive modes are not adaptive in any useful sense: because raw medians sit near
0.01–0.02 while the compared score sits near 0.24, `adaptive_quantile` returns a
threshold far *below* the whole score distribution and fires on 74% of decisions
regardless of the genome's target quantile, while `adaptive_volatility` lands above it
and behaves like a high fixed threshold. Neither mode's genome parameter has the effect
its name claims.

## The mechanism

`escalation_calibration_in_score_units` (default `False` = today's behavior) makes the
adaptive modes calibrate against a rolling window of the normalized scores they are
compared with. `AgentState.normalized_anomaly_history` records those scores at the same
memory depth as the raw window; it is runtime state, not a genome field. Fixed mode is
untouched. No subsidy, no floor, no domain concept: the change is that a genome asking
for "fire above my 90th percentile" now gets a threshold at the 90th percentile of the
quantity the decision actually uses.

Because the trait range `escalation_threshold_range` defaults to `(0.3, 0.9)` and is read
as a quantile in adaptive modes, calibrating the units also makes the *starting* range
meaningful, so the sweep includes arms that start the trait lower. That is an initial
parameter of the run, not a mid-run change, and it is varied between arms the same way
`grounded_input_fraction` was.

## Result: k stops binding, the response gate stays shut

Five seeds, 200 steps, ordinary (evolved) arm:

| Quantity | `raw_units_control` | `score_units` | `score_units` start (0.1, 0.5) | `score_units` start (0.05, 0.3) |
|---|---:|---:|---:|---:|
| Realized break-even precision | 20.00% | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0480 | 0.0473 | 0.0473 | 0.0469 |
| Correct-report rate | 12.27% | 9.89% | 12.13% | 11.41% |
| Reports per adult lifetime | 3.52 | 1.94 | 8.34 | 9.01 |
| Adult steps per agent | 12.24 | 13.58 | 16.21 | 13.55 |
| Silent-adult share | 49.07% | 51.66% | 28.88% | 25.60% |
| Attention income / agent-step | 0.0255 | 0.0247 | 0.0272 | 0.0265 |
| Fitness alignment `b` | +0.040 | +0.070 | +0.027 | +0.017 |
| Clause 1: correct-report slope / generation | -0.0008 | -0.0011 | +0.0001 | -0.0001 |
| Clause 2: parent-child offspring correlation | +0.065 | +0.034 | +0.028 | +0.076 |
| Parent-child precision correlation | +0.188 | +0.213 | +0.133 | +0.171 |

Reading:

- **Fixing the units alone reduces reporting.** `score_units` at the default trait range
  cuts reports per adult lifetime from 3.52 to 1.94, because the previously
  miscalibrated `adaptive_quantile` mode was firing on ~74% of its decisions by accident;
  once it honors its own quantile, a trait drawn from `(0.3, 0.9)` is a 30th–90th
  percentile threshold and fires far less. The mismatch was inflating report volume, not
  suppressing it.
- **With the units fixed, the trait range becomes the lever.** Starting the quantile trait
  low (0.05–0.3) more than doubles reports per adult lifetime against the control (9.01
  vs 3.52), drops the silent-adult share from 49% to 26%, and keeps adult lifespan at
  13.6 steps — reporting still pays for itself at these volumes.
- **The sample-size term is now comfortably satisfied.** `docs/heritability-measurement.md`
  put the observer's bar at ~7 verified reports per agent and
  `docs/domain-richness-requirement.md` put selection's bar near 0.9; the calibrated
  low-start arms deliver 8.3–9.0.
- **Correctness stays as heritable as lever 1 made it.** Parent-child precision
  correlation is +0.13 to +0.21 across arms, versus +0.19 for the control.
- **Neither falsification clause clears, and `b` does not improve.** Fitness alignment is
  +0.017 to +0.070 — indistinguishable from the +0.04 of lever 1 and no better in the
  arms with the most reports. Correct-report rate stays at 9.9–12.3% in every arm, above
  the 3.00% static-prior null and far below the 34.9% best clone genome, and its
  generational slope stays flat.

Twenty seeds at the best arm (`score_units`, trait start `(0.05, 0.3)`) against the
20-seed lever-1 control from `docs/false-alarm-pricing-measurement.md`, same dose and
same pricing:

| Quantity | lever 1 only | + calibrated, low start |
|---|---:|---:|
| Reports per adult lifetime | 3.33 | 9.16 |
| Correct-report rate | 10.14% | 10.75% |
| Clause 1 mean slope (seeds rising) | -0.0011 (9/20) | +0.0001 (11/20) |
| Clause 2 mean r (seeds above 0.2) | +0.096 (2/20) | +0.066 (0/20) |
| Parent-child precision correlation | +0.109 | +0.134 |

Clause 1's mean slope is no longer negative and rises in 11/20 seeds, which is a coin
flip rather than a trend; clause 2 is cleared in no seed at all, i.e. worse than the
2/20 of lever 1 despite 2.8× the reports.

## What this changes about the ordering

Levers 1 and 2 together raise reports per adult lifetime ~20× (0.45 → 9.16) and take the
silent majority from 69% to 26%, so every quantity on the *opportunity* side of the
richness requirement in `docs/domain-richness-requirement.md` is now satisfied at this
population size. What has not moved through either lever is `b`, the mapping from an
agent's measured correctness to its offspring count, which sits at +0.02 to +0.07 in the
ordinary arm no matter how many reports agents issue. Raising `k` further is therefore
predicted to be wasted effort.

That leaves the remaining two levers as the ones that can still discriminate:
lever 3 (designed-reporter arms in Scrapiron and Xylella) tests whether those domains
have any exploitable margin at all, and lever 4 (population scale toward `N_e ≈ 250`)
tests the drift term, which is the only term in the requirement that a low `b` does not
already dominate — required `N_e` scales as `1/b²`, so at `b ≈ 0.04` the honest estimate
is far above 250 and lever 4 should be read as a measurement of how `b` behaves with
scale, not as an expected pass.

Both knobs stay default-off. Turning `escalation_calibration_in_score_units` on changes
what an adaptive threshold means for every existing configuration, and no clause has been
cleared to justify that.
