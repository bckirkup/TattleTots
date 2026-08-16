# Lever 5: rationing reproduction by verified correctness

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/measure_response_gate.py \
  --steps 200 --seeds 42 43 44 45 46 --weights 0.25 0.5 1.0

uv run --no-sync --no-build python scripts/check_falsification_reliability.py \
  --steps 600 --seeds $(seq 42 61) --doses 8 \
  --break-even-precision 0.2 --score-units --threshold-range 0.05 0.3 \
  --correctness-weight 1.0
```

Generated artifacts: `docs/response-gate.md`, `docs/response-gate.json`.
Instrument: `SparseSensorScenario` (published coordinates, graded latent source, 3.00%
static-prior localization null). Every earlier lever is held fixed in every arm
(`correct_report_attention_value=8`, `false_alarm_break_even_precision=0.2`,
`escalation_calibration_in_score_units=True`, `reproduction_merit_ordering=True`,
starting `escalation_threshold` range `(0.05, 0.3)`), so arms differ only in
`reproduction_correctness_weight`.

## What the gate was keyed on

`docs/false-alarm-pricing-measurement.md`, `docs/threshold-calibration-measurement.md` and
`docs/population-scale.md` each moved the quantity they targeted — reports per adult
lifetime 0.45 → 9.2, silence 69% → 26%, effective population 60 → 500 — and each left
fitness alignment `b` (correctness → offspring) at +0.017…+0.070. That is the response
gate, and it was keyed on the wrong quantity: `reproduction._merit_ordered` ordered
eligible parents by `reproduction_sufficiency`, i.e. by reserves, and reserves are
dominated by correctness-blind information income. An agent that reported correctly was
not ordered ahead of one that merely accumulated.

`reproduction_correctness_weight` mixes rank in verified correctness into that ordering.
Both terms enter as fractional ranks within the eligible parents, so the weight is a mixing
fraction rather than an exchange rate between quantities with different units, and
correctness is shrunk toward zero by a pseudo-count of two reports so a single lucky report
does not outrank a sustained reporter. At the default `0.0` the ordering is exactly the
reserves-only one every earlier lever was measured under. This is not scaffolding: nothing
is subsidized, no agent is protected, and the cap, the solvency requirement and the
eligibility rule are untouched — only the order in which a binding cap is spent changes.

## Sweep result (200 steps, 5 seeds)

See `docs/response-gate.md` for the full table. The clause-relevant rows:

| Quantity | weight 0 | weight 0.25 | weight 0.5 | weight 1 |
|---|---:|---:|---:|---:|
| Correct-report rate | 11.41% | 15.04% | 11.83% | 12.88% |
| Reports per adult lifetime | 9.01 | 16.17 | 7.30 | 11.51 |
| Fitness alignment `b` | +0.017 | +0.058 | +0.020 | +0.066 |
| Clause 1: correct-report slope / generation | −0.0001 | +0.0026 | +0.0010 | +0.0023 |
| Clause 2: parent-child offspring correlation | +0.076 | −0.030 | +0.083 | +0.054 |
| Parent-child precision correlation | +0.171 | +0.123 | +0.138 | +0.168 |

Keying the gate on correctness turns the clause-1 slope from flat to positive and roughly
quadruples `b`, but the response is not monotone in the weight at five seeds, so the sweep
alone does not settle it.

## Reliability across seeds

`check_falsification_reliability.py`, 20 seeds, ordinary arm, all earlier levers fixed:

| Run | Correct-report rate | Reports/adult | Clause 1 slope | Clause 1 seeds rising | Clause 2 mean | Clause 2 seeds cleared |
|---|---:|---:|---:|---:|---:|---:|
| 200 steps, weight 0 | 10.75% | 9.16 | +0.0001 | 11/20 | +0.066 | 0/20 |
| 200 steps, weight 0.25 | 11.28% | 11.43 | +0.0016 | 14/20 | +0.027 | 0/20 |
| 200 steps, weight 1 | 11.09% | 8.49 | +0.0018 | 13/20 | +0.073 | 0/20 |
| 600 steps, weight 0, seeds 42–61 | 13.21% | 39.00 | +0.0015 | 14/20 | −0.060 | 0/20 |
| 600 steps, weight 1, seeds 42–61 | 14.32% | 37.26 | +0.0034 | 18/20 | −0.043 | 0/20 |
| 600 steps, weight 0, seeds 101–120 | 13.23% | 40.48 | +0.0007 | 15/20 | −0.049 | 0/20 |
| 600 steps, weight 1, seeds 101–120 | 14.60% | 36.23 | +0.0025 | 16/20 | −0.059 | 0/20 |

At 200 steps a correctness-keyed gate raises the mean slope more than tenfold but rises in
only about two thirds of seeds — better than a coin flip and not reliable. At 600 steps,
where each run spans enough generations for a per-generation slope to be estimated against
less noise, the correctness-keyed gate rises in 34/40 seeds (85%) at +0.0025…+0.0034 per
generation. The seeds 101–120 block is a holdout: it was run after the 42–61 result to check
that the effect is not a property of the seed set the lever was developed on, and the slope
ratio replicates (3.5x the control there, 2.3x here).

The honest reading of the control column is that run length carries part of this. The
reserves-only ordering also rises at 600 steps (29/40 seeds, +0.0007…+0.0015), because more
generations means less noise in the slope *and* more time for whatever weak selection
exists to act. What the gate key contributes is a 2–3x larger slope and a 1 pp higher
realized correct-report rate at every seed block measured, not the entire rise.

## Why clause 2 does not move

Clause 2 asks for parent–child *reproductive* correlation above ~0.2, and the ordering
lever cannot supply it, because the gate has very little differential to distribute:

| Quantity | weight 0 | weight 1 |
|---|---:|---:|
| Eligible share of adults per step | 65.2% | 70.2% |
| Steps where the population cap binds | 36.2% | 37.9% |
| Mean offspring, adults with a correct report | 1.312 | 1.265 |
| Mean offspring, adults reporting but never correct | 1.255 | 1.237 |
| Mean offspring, silent adults | 1.285 | 1.267 |

Two thirds of adults are reproductively eligible on a given step and the cap binds on
about a third of steps, so ordering changes who reproduces *first*, not how many offspring
a lineage gets: the spread between a correct reporter and a silent agent is ~0.03 offspring,
about 2%. Selection on ordering is real but its bandwidth is set by how often the cap binds
and how few eligible agents it excludes — a scarcity property of the reproduction rule, not
of the ordering key. That is the quantity to attack next if clause 2 matters: reproductive
excess, i.e. the share of eligible agents the environment can actually afford, is the term
that converts an ordering advantage into a heritable reproductive one.

Correctness heritability is unchanged by the lever (+0.12…+0.17 across all arms and seed
blocks), which is what it should be: the lever changes who reproduces, not how faithfully a
genome transmits.

## Verdict against the falsification test

- **Clause 1 (correct-report rate rises over generations at fixed initial parameters):**
  met at 600 steps with a correctness-keyed gate — +0.0025…+0.0034/generation, rising in
  34/40 seeds across two independent seed blocks, at a 13–15% realized rate against a
  3.00% static-prior null. This is the first arm measured in this series that satisfies a
  clause reliably rather than at chance. Attribution is partial: the reserves-only control
  at the same run length rises in 29/40 seeds at a 2–3x smaller slope, so run length
  supplies part of the effect and the gate key supplies the rest.
- **Clause 2 (parent–child reproductive correlation reliably above ~0.2):** not met,
  0/20 seeds in every arm and every seed block, for the reproductive-excess reason above.
