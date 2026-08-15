# Cross-domain measurement of the grounded-input fix

Date: 2026-08-15

This supersedes the `gaussian_shift` conclusions in
[`docs/initiation.md`](initiation.md) and [`docs/ceiling-test.md`](ceiling-test.md).
Every number here comes from a modeled instrument with a non-vacuous
localization null, measured after the grounded-input knobs landed in
TattleTots #55 (`grounded_input_fraction`, `grounded_attractiveness_multiplier`,
`max_input_streams`). No scaffolding — no subsidies, grace periods, juvenile
discounts, or population floors — was added in any domain.

## The mechanism under test

`select_input_streams` previously picked three streams by softmax over
`structured_variance` with no preference for `StreamType.RAW`, so agents were fed
mostly peer residual exhaust. `grounded_input_fraction` reserves
`ceil(fraction × slots)` slots for raw grounded streams, capped by how many raw
streams the domain publishes. Defaults (`0.0`, multiplier `1.0`) reproduce the
previous behavior, including RNG consumption.

## Cross-domain result

| Domain | Instrument null (static prior) | Grounded-yield share, 0 → post-fix | Precision, 0 → post-fix | Correct-report trend | Parent–child repro corr, 0 → post-fix |
|---|---:|---|---|---|---|
| TattleTots (SparseSensor, 5 seeds, 200 steps) | 3.00% | 4.61% → 52.97% | 3.09% → **12.80%** (clears null) | **rises** within run (−2.23% → +0.87% drift) | 0.092 → 0.114 |
| Coral Key (AIS/SAR, 20 seeds, 200 epochs, cap 48) | 14.84% | 1.24% → 100% | ordinary 2.07% → **15.84%** (clears null marginally, at fraction 0.67); designed 32.10% → 57.61% | not measured as within-run drift | 0.019 → 0.134 |
| Scrapiron (fire, OPIR ablated, 5 seeds) | 35.6% | 10.3% → 100% | 14.2% → 18.0% (beats 8.0% chance null, **loses** to static-prior null) | agent-only detection **falls** (2.48% → 1.48%) | ≈0 (−0.036 → ≈0) |
| Xylella (grain, pests frozen, 10 seeds, 400 steps) | 54.95% | 11.0% → 100% | 35.0% → 50.6% (never clears null) | **falls** (Δ −0.315 → −0.492) | −0.014 → 0.048 (pest reference: **+0.222**) |

Sources: `docs/ceiling-measurement.md` (this repo),
[Coral PR #29](https://github.com/bckirkup/Coral_Key_in_Three_Hour_Epochs/pull/29),
[Scrapiron PR #27](https://github.com/bckirkup/Scrapiron_and_the_Bear/pull/27),
[Xylella PR #29](https://github.com/bckirkup/Xylella_SPQR/pull/29). Each domain
emits `tattletots.output_schema.SimulationOutput` per run.

## What the four measurements agree on

1. **Input starvation was real and is fixed.** Grounded-yield share rises from
   1–11% to 53–100% in every domain, and the engine's
   `grounded_yield_share_below_minimum` degeneracy flag clears. Coral's
   hand-designed reporter, which previously saw AIS/SAR evidence on 1.61% of
   adult steps, now sees it on 78.72% at fraction 0.67.
2. **Reserved slots are the operative lever, not weighting.** Coral measured the
   attractiveness multiplier in isolation (fraction 0.0, multiplier 3.0):
   evidence rate moved 1.61% → 2.08% and ordinary precision 2.07% → 2.31%. The
   reservation mechanism is what changes the input diet.
3. **Reporting quality improves wherever grounded evidence is decodable.**
   Precision rises in all four domains. It clears the static-prior null in
   TattleTots and (marginally) in Coral; it does not clear it in Scrapiron or
   Xylella, whose instruments have much higher static priors (35.6% and 54.95%),
   i.e. a well-placed constant guess is already strong there.
4. **Heritable reproductive success is still missing.** Parent–child
   reproductive correlation stays at 0.05–0.13 everywhere, against the ~0.2
   falsification bar. Xylella is the sharpest control: in the same runs, the
   pest side reaches +0.207 to +0.222 while the detector side sits at ≈0. The
   deficit is in the detector-side selection loop, not in the measurement.

## Verdict against the falsification test

The falsification test from `docs/initiation.md` has two clauses.

- *Correct-report rate rises over a run without changing initial parameters, on
  a real instrument with a non-vacuous null.* **Met** on SparseSensor
  (3.09% → 12.80%, drift +0.87%, static-prior null 3.00%). Not met in Scrapiron
  or Xylella, where the within-run trend is negative and becomes more negative
  with more grounded input.
- *Parent–child reproductive correlation reliably above ~0.2.* **Not met**
  anywhere (best: Coral 0.134).

So the superseded claim that "competence is not expressible in genome space" is
refuted — it was an artifact of `gaussian_shift`'s vacuous null plus input
starvation. The remaining failure is narrower and better localized: correctness
does not convert into differential reproduction. Xylella's frozen-pest contrast
shows this is a property of the detector-side economy, since the same engine
transmits reproductive success on the pest side in the same runs.

## Honest caveats

- Every domain measured the post-fix arms against an editable install of the
  TattleTots branch; the domain lockfiles still pin the previous rev, so those
  runs are not reproducible from the lockfiles alone until the pins are bumped.
- Scrapiron and Xylella saturate at their population caps in every arm, so
  extinction and reproductive selection are not exercised; their near-zero
  reproductive correlations are consistent with, but not proof of, a missing
  gradient.
- Coral's baseline invasion arm still pools few designed reports (31 over 20
  seeds, 13 seeds with none), so its baseline precision is lineage-driven. The
  post-fix arms do not have this problem (3,114 reports).
- The fraction knob saturates once reservations exhaust the domain's raw
  streams. `SparseSensorScenario` publishes exactly one raw stream, so fractions
  0.34, 0.67 and 1.0 are identical there by construction.
- Scrapiron's result is a genuine negative: with the OPIR backstop ablated,
  agent-only detection *falls* as grounded input rises (2.48% → 1.48%), even
  though precision improves. Grounded input buys better reports, not more of
  them, when attention is the binding constraint.

## What to do next

Items 1 and 2 of `docs/initiation.md` ("make competence expressible", "measure
initiation before engineering it") are now done and reproducible. The open work
is items 3 and 4 — couple the currencies so attention income is earned by
verified-correct reports, and price reproduction in the scarce currency — with
Xylella's pest loop as the reference shape and the parent–child reproductive
correlation as the acceptance metric. Scaffolding stays the last lever.
