# Lever 1: repricing false alarms against reachable precision

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/measure_false_alarm_pricing.py
uv run --no-sync --no-build python scripts/check_falsification_reliability.py \
  --doses 8.0 --break-even-precision 0.2
```

Generated artifacts: `docs/false-alarm-pricing.md`, `docs/false-alarm-pricing.json`.
Instrument: `SparseSensorScenario` (published coordinates, graded latent source, 3.00%
static-prior localization null). Every arm below shares the same initial parameters;
only `SimulationConfig.false_alarm_break_even_precision` differs.

## What was mispriced

`docs/reporting-opportunity-measurement.md` found that evidence arrives (93% of adult
steps carry grounded yield) but 79.5% of adults never report, because reporting was
priced against a precision no evolved agent can reach: a flat `false_alarm_penalty` of
0.3 against ~0.02 attention income per agent-step put break-even precision near 80%
against a decoder ceiling of ~16%. Escalating *worsened* attention drift, so silence was
the evolved optimum and the fitness alignment `b` of
`docs/domain-richness-requirement.md` stayed near zero for structural reasons rather
than for want of heritable variance.

## The mechanism

`false_alarm_break_even_precision` (default `None` = the flat penalty, i.e. today's
behavior) prices one false alarm as

```text
price = value_per_correct_report * p / (1 - p)
```

which is exactly the charge that makes the expected attention return on a report,
`p * value - (1 - p) * price`, change sign at precision `p`. `value_per_correct_report`
is the agent's own attention allocation times `correct_report_attention_value`, so the
price tracks the agent's standing with users instead of a global constant, and the
mechanism stays domain-agnostic: no domain concept, no subsidy, no floor, and reporting
below the target still costs the agent attention.

The knob requires `correct_report_attention_value > 0`, since that term supplies the
value side of the trade; without it the flat penalty remains in force.

## Result: reporting opportunity is unblocked, neither clause clears

Five seeds, 200 steps, ordinary (evolved) arm, `correct_report_attention_value=8`,
merit-ordered reproduction on in every arm:

| Quantity | `flat_penalty` | `break_even_0.4` | `break_even_0.2` | `break_even_0.1` | `break_even_0.05` |
|---|---|---|---|---|---|
| Realized break-even precision | 63.63% | 40.00% | 20.00% | 10.00% | 5.00% |
| Attention charged per false alarm | 0.2711 | 0.1305 | 0.0480 | 0.0224 | 0.0102 |
| Correct-report rate | 10.40% | 10.48% | 12.27% | 11.24% | 11.42% |
| Reports per adult lifetime | 0.45 | 4.11 | 3.52 | 3.39 | 3.13 |
| Adult steps per agent | 7.03 | 12.92 | 12.24 | 13.24 | 12.27 |
| Silent-adult share | 69.11% | 50.86% | 49.07% | 50.09% | 52.39% |
| Fitness alignment `b` | +0.044 | +0.049 | +0.040 | +0.039 | +0.060 |
| Clause 1: correct-report slope / generation | -0.0002 | -0.0018 | -0.0008 | -0.0013 | -0.0006 |
| Clause 2: parent-child offspring correlation | +0.127 | +0.052 | +0.065 | +0.043 | +0.085 |
| Parent-child precision correlation | +0.041 | +0.194 | +0.188 | +0.086 | +0.045 |

Twenty seeds at `break_even_precision=0.2` against the same flat-penalty control:

| Quantity | flat penalty | priced at 0.2 |
|---|---|---|
| Reports per adult lifetime | 0.46 | 3.33 |
| Correct-report rate | 10.72% | 10.14% |
| Clause 1 mean slope (seeds rising) | +0.0000 (12/20) | -0.0011 (9/20) |
| Clause 2 mean r (seeds above 0.2) | +0.124 (1/20) | +0.096 (2/20) |
| Parent-child precision correlation | +0.077 | +0.109 |

Reading:

- **The pricing does what it was derived to do.** Realized break-even precision follows
  the target exactly, and the charge per false alarm falls 5–27× below the flat penalty.
- **Reporting opportunity is no longer the binding throttle.** Reports per adult
  lifetime rise 7–9× (0.45 → 3.1–4.1), adult lifespan nearly doubles (7.0 → 12.2–12.9
  adult steps, i.e. reporting now pays for itself instead of shortening life), and the
  silent-adult majority drops from 69% to ~50%.
- **Correctness becomes measurably heritable.** Parent-child precision correlation rises
  from +0.041 to +0.194 at the two mildest useful targets — consistent with the
  attenuation model in `docs/domain-richness-requirement.md`, where reliability rises
  with reports per agent. The 20-seed run puts it at +0.109 vs +0.077, so the size of
  the gain is seed-sensitive even though the direction is not.
- **Neither falsification clause clears.** Clause 1 (within-run rise in correct-report
  rate at fixed parameters) is flat to slightly negative and rises in only 9/20 seeds;
  clause 2 (parent-child reproductive correlation above ~0.2) is met in 2/20 seeds
  against 1/20 for the flat control. Correct-report rate itself does not move: ~10–12%
  in every arm, above the 3.00% static-prior null but far below the 34.9% best clone
  genome.
- **`b` did not improve.** Fitness alignment stays at +0.04 in the ordinary arm across
  every target. This is the measurement lever 1 was expected to move, and it did not:
  cheaper false alarms buy *more* reports, not reports whose correctness converts into
  offspring. (The +0.44 alignment quoted in `docs/payoff-fix-measurement.md` was
  measured with an oracle invasion supplying correctness spread; the ordinary-arm value
  is what selection actually sees.)

## What this changes about the ordering

Lever 1 removes the throttle it was aimed at and leaves the response gate open, so the
diagnosis moves down the list in `docs/domain-richness-requirement.md`: with `k` now
above 3 reports per agent, the requirement `k*` is satisfied at any plausible variance
estimate, and what remains binding is not sample size but the mapping from measured
precision to reproductive success — the same quantity `b` — plus the escalation
threshold, which still sits above most of the anomaly distribution and is what lever 2
addresses. Population scale (lever 4) is not implicated: required `N_e` falls with `k`,
and `k` just rose 7×.

The knob stays default-off. Turning it on is a claim about the reward geometry a domain
should present, and no clause has yet been cleared to justify making it the default.
