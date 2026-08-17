# Lever 6: reproductive excess, and why per-step scarcity is not the missing term

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/measure_reproductive_excess.py \
  --steps 600 --seeds $(seq 42 61) --shares 0.5 0.25 0.1

uv run --no-sync --no-build python scripts/measure_reproductive_excess.py \
  --steps 600 --seeds $(seq 101 120) --shares 0.1 --no-write
```

Generated artifacts: `docs/reproductive-excess.md`, `docs/reproductive-excess.json`.
Instrument: `SparseSensorScenario` (3.00% static-prior localization null). Every earlier
lever is held fixed in every arm (`correct_report_attention_value=8`,
`false_alarm_break_even_precision=0.2`, `escalation_calibration_in_score_units=True`,
`reproduction_merit_ordering=True`, `reproduction_correctness_weight=1.0`, starting
`escalation_threshold` range `(0.05, 0.3)`), so arms differ only in
`reproduction_recruitment_share`.

## What was predicted, and what the prediction got wrong

`docs/response-gate-measurement.md` closed by naming reproductive excess as the term that
converts an ordering advantage into a reproductive one: with two thirds of adults eligible
each step and the cap binding on about a third of steps, the spread between a correct
reporter and a silent adult was ~0.03 offspring. The predicted fix was to make eligibility
scarce so rank decides who reproduces *at all*.

`SimulationConfig.reproduction_recruitment_share` implements that as a limit on recruitment
rather than a payment: below `1.0`, only that share of a step's eligible parents may recruit,
and nobody is subsidized, protected, or given a floor. At the default `1.0` the behavior is
exactly the previous one.

Measuring it refutes the premise. Reproductive excess is *already* extreme at the default,
because the binding limit is the room left under the population cap, not the eligibility
rule: with the population pinned at 60, eligible parents outnumber affordable recruits
31.7:1, and eligible parents outnumber slots on 92.5% of steps. There is no per-step
scarcity left to add.

## Sweep result (600 steps, seeds 42–61, 20 seeds)

| Quantity | share 1 (control) | share 0.5 | share 0.25 | share 0.1 |
|---|---:|---:|---:|---:|
| Reproductive excess (eligible / slot) | 31.67 | 50.33 | 64.08 | 76.99 |
| Slot-limited step share | 92.50% | 100.00% | 100.00% | 100.00% |
| Mean offspring per adult | 1.27 | 1.26 | 1.28 | 1.27 |
| Opportunity for selection `I` | 0.875 | 0.796 | 0.874 | 1.037 |
| Correct-report rate | 14.32% | 13.95% | 14.20% | 14.40% |
| Reports per adult lifetime | 37.26 | 38.65 | 36.14 | 37.25 |
| Adult steps per agent | 51.94 | 54.09 | 50.55 | 54.95 |
| Silent-adult share | 14.40% | 14.02% | 12.84% | 4.63% |
| Fitness alignment `b` | −0.040 | −0.054 | −0.019 | −0.024 |
| Clause 1: correct-report slope / generation | +0.0034 | +0.0030 | +0.0034 | +0.0033 |
| Clause 2: parent–child offspring correlation | −0.043 | −0.038 | −0.023 | −0.025 |
| Parent–child precision correlation | +0.152 | +0.156 | +0.134 | +0.146 |
| Final population | 60.0 | 60.0 | 59.9 | 60.0 |

Cutting the recruitment share tenfold raises the measured excess (31.7 → 77.0 eligible per
slot) and changes nothing that selection consumes: mean offspring per adult is 1.26–1.28 in
every arm, the opportunity for selection stays at 0.8–1.0, and clause 2 stays slightly
negative. Clause 1 remains met in every arm at the same slope as the control, so the lever
neither helps nor harms the one clause that does clear.

## Why tightening per-step scarcity cannot work here

Total lineage output is set by mortality, not by the recruitment rule. The population sits
at the cap in every arm, so births equal deaths whatever the share is; restricting how many
parents may recruit *per step* only spreads the same number of births over more steps.

Losing a step's competition also costs an adult nothing, because eligibility is not consumed:
an adult that is out-ranked stays eligible and competes again next step, and it lives ~52
adult steps. At 31.7 eligible per slot an eligible adult wins roughly one contest per 32
steps, i.e. ~1.6 over its adult life — which is the 1.27 offspring actually measured. Rank
therefore reorders a queue that everyone reaches the front of, so it shifts the *timing* of
reproduction, not lifetime output, and lifetime output is what clause 2 correlates.

That is a general statement about the reproduction rule rather than about a share value: any
per-step ordering over a persistently-eligible population with long adult lives converts to
differential lineage output only in proportion to the adults that die while still waiting.
The quantity to attack next is that one — how strongly mortality is coupled to rank, or
equivalently whether an adult's rank persists long enough to matter over a lifetime — not
how scarce a single step's recruitment is.

## Holdout block (600 steps, seeds 101–120)

Run after the 42–61 result, on the control and the tightest share only, to check that the
null result is not a property of the seed set the lever was developed on:

| Quantity | share 1 (control) | share 0.1 |
|---|---:|---:|
| Reproductive excess (eligible / slot) | 29.37 | 76.39 |
| Slot-limited step share | 90.75% | 100.00% |
| Mean offspring per adult | 1.26 | 1.27 |
| Opportunity for selection `I` | 0.754 | 1.038 |
| Correct-report rate | 14.60% | 14.49% |
| Silent-adult share | 15.91% | 4.22% |
| Clause 1: correct-report slope / generation | +0.0025 | +0.0029 |
| Clause 2: parent–child offspring correlation | −0.059 | −0.030 |

The holdout replicates every number that matters: excess rises 2.6x, offspring per adult
does not move, and clause 2 stays negative.

One real side effect is worth recording because it is not what the lever was aimed at:
tightening the share cuts the silent-adult share from ~15% to ~4% and raises reports per
adult lifetime (36.2 → 40.8 on the holdout block) while adults live slightly longer
(49.6 → 53.2 adult steps) — most likely because reproduction spends reserves and an adult
that loses a contest keeps them, though that mechanism is inferred, not measured. More
reporting at an unchanged correct-report rate is more evidence about the same precision, so
it moves the parent–child *precision* correlation not at all (+0.116 → +0.125), which is
consistent with the sampling-noise account in `docs/heritability-measurement.md`.

## Verdict against the falsification test

- **Clause 1 (correct-report rate rises over generations at fixed initial parameters):**
  still met, unchanged by this lever — +0.0030…+0.0034/generation in all four arms, at a
  14% realized rate against a 3.00% static-prior null. The lever contributes nothing to it.
- **Clause 2 (parent–child reproductive correlation reliably above ~0.2):** not met, and
  not for the reason the previous lever predicted. Per-step reproductive excess is already
  31.7:1 at the default and raising it to 77:1 leaves offspring per adult unchanged, so
  scarcity of recruitment is not the missing term. Eligibility that is never consumed plus
  long adult lives is.

## Status of the knob

`reproduction_recruitment_share` stays in the engine at its default `1.0` (no behavioral
change) as the instrument this result rests on: it is the arm that shows more per-step
scarcity is not purchasable improvement. It is domain-agnostic, it is a limit rather than a
payment, and the population cap, the solvency requirement and the eligibility rule are
untouched by it.
