# Lever 7: rank-coupled mortality, and why clause 2 cannot be reached from here

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/measure_rank_mortality.py \
  --steps 600 --seeds $(seq 42 61) --budget-scales 0.5 0.25

uv run --no-sync --no-build python scripts/measure_rank_mortality.py \
  --steps 600 --seeds $(seq 101 120) --budget-scales 0.25 --no-write
```

Generated artifacts: `docs/rank-mortality.md`, `docs/rank-mortality.json`.
Instrument: `SparseSensorScenario` (3.00% static-prior localization null). Every earlier lever
is held fixed in every arm (`correct_report_attention_value=8`,
`false_alarm_break_even_precision=0.2`, `escalation_calibration_in_score_units=True`,
`reproduction_merit_ordering=True`, `reproduction_correctness_weight=1.0`,
`reproduction_recruitment_share=1.0`, starting `escalation_threshold` range `(0.05, 0.3)`), so
arms differ only in the scale applied to each user's attention budget — the currency whose
insolvency causes essentially every death. No engine mechanics were changed for this
measurement: the sweep is telemetry plus an environmental scarcity dial, with no subsidy,
grace period, juvenile discount, or population floor anywhere.

## What was predicted

`docs/reproductive-excess-measurement.md` closed by naming rank-coupled mortality as the one
term left for clause 2: per-step scarcity is already 31.7 eligible parents per affordable
recruit, eligibility is not consumed when an adult loses a contest, and an adult lives ~52
adult steps, so ranking was thought to reorder a queue that everyone eventually reaches the
front of. The prediction was that rank only becomes consequential if a low-ranked adult
*dies while still waiting*, and that raising mortality pressure would produce that coupling
and with it a parent–child offspring correlation above ~0.2.

## What the telemetry measures

`PayoffLedger` now records, per adult step, each living adult's fractional rank in the same
`verified_correctness` quantity the reproduction ordering sorts on (the engine helpers are
shared rather than re-derived, so telemetry and the gate cannot drift apart). From those
histories it reports rank persistence (early-life vs late-life rank), rank against adult
lifespan and against offspring, lifespan against offspring, and the childless-vs-parent
contrast in rank, adult steps, and eligible steps.

## Sweep result (600 steps, seeds 42–61, 20 seeds)

| Quantity | `budget_scale_1` (control) | `budget_scale_0.5` | `budget_scale_0.25` |
|---|---:|---:|---:|
| Died before the run ended (adults) | 90.04% | 94.46% | 97.18% |
| Childless adult share | 24.97% | 15.94% | 8.31% |
| Rank persistence (early vs late life) | +0.667 | +0.634 | +0.588 |
| Rank → adult lifespan | +0.054 | +0.090 | +0.233 |
| Rank → offspring | +0.359 | +0.291 | +0.140 |
| Lifespan → offspring | −0.116 | −0.128 | −0.076 |
| Childless mean rank vs parent mean rank | 0.413 / 0.507 | 0.419 / 0.495 | 0.460 / 0.477 |
| Childless eligible steps vs parents' | 109.6 / 32.1 | 51.9 / 17.0 | 13.0 / 8.9 |
| Childless adult steps vs parents' | 111.0 / 32.6 | 54.3 / 17.5 | 17.4 / 9.5 |
| Childless adults never eligible | 9.95% | 14.99% | 26.43% |
| Reports per adult lifetime | 37.26 | 16.78 | 6.13 |
| Adult steps per agent | 51.94 | 23.66 | 10.16 |
| Correct-report rate | 14.32% | 11.72% | 8.51% |
| Opportunity for selection `I` | 0.875 | 0.538 | 0.300 |
| Clause 1: correct-report slope / generation | +0.0034 | +0.0007 | −0.0003 |
| Clause 1: seeds rising | 18/20 | 15/20 | 6/20 |
| Clause 2: parent–child offspring correlation | −0.043 | +0.022 | +0.053 |
| Clause 2: seeds above 0.2 | 0/20 | 0/20 | 0/20 |
| Parent–child precision correlation | +0.152 | +0.165 | +0.129 |
| Final population | 60.0 | 59.9 | 58.1 |

Holdout block (seeds 101–120, control and `budget_scale_0.25`) reproduces every sign and
magnitude: rank persistence +0.683/+0.584, rank → lifespan +0.057/+0.220, rank → offspring
+0.332/+0.125, clause 1 +0.0025/−0.0003, clause 2 −0.059/+0.057.

## The hypothesis is refuted, and by its own mechanism

Low-ranked adults do not die while waiting. They *outlive* the adults that reproduce, and by a
factor of three: a childless adult accumulates 111 adult steps and 110 eligible steps against
a parent's 33 and 32, and only 10% of childless adults were never eligible at all. So the
queue is not one that everyone reaches the front of — a chronically eligible, lower-ranked
adult (mean rank 0.413 vs 0.507) is passed over for its entire life. Rank already produces
reproductive exclusion without any mortality coupling, which is why rank → offspring is
+0.359 at the control, the strongest fitness alignment measured anywhere in this series.

Raising mortality pressure does create the predicted coupling — rank → adult lifespan goes
+0.054 → +0.233 — and it makes things worse on every axis that matters. It shortens adult life
5× (52 → 10 adult steps), cuts reporting opportunity 6× (37 → 6 reports per lifetime), drops
the correct-report rate below its own control (14.3% → 8.5%), halves rank → offspring
(+0.359 → +0.140) because there is no longer time to accumulate a reliable correctness record,
and takes clause 1 from 18/20 seeds rising to 6/20. Clause 2 rises from −0.043 to +0.053 — a
change of the same size as its seed-to-seed noise, and still 4× short of the bar.

Lifespan is also the wrong currency to buy offspring with: lifespan → offspring is *negative*
in every arm (−0.08…−0.13). Reproduction concentrates in early adult life, so a mechanism that
works by shortening a low-ranked adult's life cannot redistribute lifetime output.

## Why clause 2 is unreachable, arithmetically

Parent–child offspring correlation is a *product*, not an independent quantity. With
correctness rank the trait selection acts on, an offspring count that responds to rank with
correlation `b`, and rank heritability `h`, the parent–child offspring correlation is
approximately `b² · h`. The measured components give:

| Arm | `b` (rank → offspring) | `h` (parent–child precision) | predicted `b² · h` | measured clause 2 |
|---|---:|---:|---:|---:|
| `budget_scale_1` | +0.359 | +0.152 | 0.020 | −0.043 |
| `budget_scale_0.5` | +0.291 | +0.165 | 0.014 | +0.022 |
| `budget_scale_0.25` | +0.140 | +0.129 | 0.003 | +0.053 |

The predicted ceiling is ~0.02 — an order of magnitude below the 0.2 bar — and the measured
values are that ceiling within noise. Reaching 0.2 at the measured heritability of ~0.15 would
require `b ≈ 1.15`, which is impossible; at a perfect `h = 1.0` it would still require
`b ≈ 0.45`, above anything measured. Squared-correlation shares agree: rank accounts for 13.2%
of offspring variance at the control and lifespan for 2.1%, so the offspring channel is
dominated by neither.

This makes clause 2 a stricter statement than it reads as. Clause 1 asks whether correctness
rises within a run, and it does. Clause 2, as a parent–child correlation of *offspring counts*,
asks the run to transmit a noisy count through a single generation, and it is bounded by the
square of the alignment times the heritability. The binding term it exposes is heritability of
precision, which sits at +0.13…+0.17 against the ICC of 0.63 that clone monocultures measured
in `docs/heritability-measurement.md` — i.e. an individual's realized precision is still mostly
sampling noise even at 37 reports per lifetime, so what a parent transmits is its genome and
what clause 2 scores is its luck.

## Where this leaves the series

- Fitness alignment is no longer the gap: rank → offspring is +0.359, and low-ranked adults are
  reproductively excluded for life rather than merely delayed.
- Mortality pressure is not a lever but a cost: every increment trades reporting opportunity and
  clause 1 for a coupling that clause 2 barely registers.
- The measurable next term is the heritability side of the product — closing the gap between
  realized precision (`h ≈ 0.15`) and genomic precision (ICC 0.63), either by scoring parents on
  their genome-level record (more reports per adult, longer adult life, or pooled lineage
  evidence) or by accepting a clause-2 statement in the heritable quantity (parent–child
  *precision* correlation, already +0.15) rather than in offspring counts.
