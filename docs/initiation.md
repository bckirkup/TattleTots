# Initiation: the population goes self-sustaining before it is ever any good

Status: analysis, no code changes. Re-baselined on reproducible `main` after
TattleTots #22. Measured across TattleTots, Coral Key, Scrapiron (fire), Xylella
(grain), and domain-runner.

## The dilemma as posed

Every evolutionary mechanism has a start-up problem: you need enough function for
selection to have something to select, but the cheapest way to get that function is
to optimize it by hand — and then the mechanism is decoration on top of your hand
work. The question is what the initiation problem actually *is* in this codebase,
per domain, rather than what it is in general.

## What it actually is here

It is not a cold-start deficit. A cold-started random population is immediately
self-sustaining: in 60 reproducible runs (`gaussian_shift`, 200 steps,
`initial_population=20`, `max_population=60`, mutation 0.1, recombination 0.3,
false-alarm penalty 0.4, subsidy 0.1) the first birth happened at step **1** in
**60/60** runs, no run went extinct, and ~17 descendant lineages were alive at
step 200. The earlier figures shifted because seeded runs were not reproducible
before TattleTots #22; the ecology mechanics did not change.

It is also not that nothing is heritable. The genome is genuinely mutable and
load-bearing: `compression_type`, `n_components`, `sensing_strategy`,
`working_dim`, `escalation_threshold`, and the cost traits all change what an
agent computes, what it reports, and what it pays
(`engine/world.py` `_init_agent_model`/`_compress`/`_maybe_escalate`).

The problem is that **the loop closes on reproduction before it closes on
function**, so the population is sustained by something other than doing its job.
The deterministic baseline also sharpens the precision result: reports are mostly
false alarms, but precision above the empirical chance null is achievable in some
cohorts rather than absent:

| Measured | Value |
|---|---|
| Reports that were false alarms | 232.65 of 233.50 per run in Arm B / 206.65 of 209.75 in Arm C (**~99%**) |
| Correct reports per run | **0.85** (Arm B) / **3.10** (Arm C) |
| `precision_not_above_chance` | **16/20** Arm A, **15/20** Arm B, **5/20** Arm C |
| `grounded_yield_share_below_minimum` | **20/20** in every arm |
| Original agents that reproduced at least once | **387 / 400** in Arm A |
| Original agents alive at step 25 | **0 / 400**, in every arm |
| Parent–child lifetime-reproduction correlation | r = **0.009** (p = **0.31**, n = **12,510**) |
| Info energy, first 25 steps | mean **+6.66**, cross-agent SD 12.95 |
| Attention energy, first 25 steps | mean **-1.35**, cross-agent SD **0.33** |

Read together: reproduction remains near-universal, death is universal and
undifferentiated, reproductive success does not transmit, and grounded yield
fails in every arm while the ecology reports a healthy sustained population.
Precision is not absent: some cohorts beat chance, especially with the environment
fixed in Arm C. The sharper finding is that reproduction is funded by compressing
other agents' exhaust rather than by being grounded in the world. Selection is
running; it is running on almost nothing that matters.

Two hypotheses that this measurement **refutes**, both of which we would otherwise
have carried forward:

* *Threshold miscalibration is the killer.* 368 of 400 original agents never once
  crossed their own escalation threshold, yet 98.6% of them still reproduced, and
  death was 100% in every band (mute / middle / screamer). The middle band's
  reproduction rate was only 40% (n = 10), while mutes were 98.6% and screamers
  90.9%; that small middle-band result is suggestive rather than robust.
  Threshold-vs-anomaly scale mismatch still does not explain the universal
  mortality.
* *Attention is flat because of the equal-share fallback.* The fallback in
  `allocate_attention` fired **0** times in 3,000 allocations; every allocation
  took the weighted trust×relevance branch. Attention is flat *despite* being
  weighted.

## Why (three structural causes, in order of severity)

**1. The two currencies do not share a carrying capacity.** Reproduction is gated on
information energy, which is unbounded — it is minted per agent per step by
compressing whatever is in front of it. Survival is gated on attention energy,
which is fixed-sum: 2 users × `attention_budget=1.0`. Mean `maintenance_cost` is
~0.06, so the attention economy supports ~33 agents; the information economy pays
for breeding to `max_population=60`. The population therefore *always* overshoots
into universal attention starvation, and starvation that hits everyone is not
selection. With a roughly 99% false-alarm rate, `false_alarm_penalty` burns roughly a
quarter of the entire attention economy on top of that.

**2. Information yield pays for compressing anything, including peers' noise.**
Yield is variance-explained on the input, discounted by nothing that knows whether
the input matters. Agents publish residual streams that other agents consume and
get paid to compress (`_publish_output_stream` → `_attach_trophic_inputs`), plus a
flat `subsidy_rate` per downstream consumer. A trophic chain can therefore be
solvent while grounded in nothing — the currency that funds reproduction never
touches ground truth.

**3. Reproduction is cheaper than function.** Offspring cost is
`reproduction_threshold / 8` per parent (~0.28) against ~6.4 of information energy
accrued in 25 steps. Nothing about being right is required to pay it. Consistently,
the larger within-run selection differentials remain on reproduction-adjacent traits:
lower `reproduction_threshold` (d = -0.40) and lower `compute_cost` (d = -0.61),
not a reliable inherited reproductive advantage. The old d = -0.57 and -0.70
figures came from the nondeterministic baseline.

The variance decomposition must be read with one qualification. Relative to the
earlier run, the SD(C)/SD(B) ratios moved from **1.02 to 1.36** for total reports,
**1.00 to 1.33** for false alarms, **3.24 to 1.89** for correct reports, and
**0.56 to 1.03** for descendant lineages. Those shifts are why the earlier
figures cannot be carried forward as fixed evidence.

More importantly, Arm B does not isolate environment variation. Its genome cohort
is fixed, but varying the world seed also varies the agent, stream, and user IDs;
those IDs feed the stable digest-based simulation choices. Arm B therefore measures
environment **plus identity**. This confound is not new: before #22 those IDs were
random on every run, so Arm B never isolated the environment. What changed is that
the confound is now deterministic and visible. The B/C ratios are not a clean
genome-versus-environment decomposition in either baseline. Any future claim
resting on them needs an arm that holds identities fixed while varying the
environment.

## The general condition (the forest)

For selection to bootstrap function, the currency that decides who persists has to
satisfy all three of: **(a)** mean near break-even, so the population is neither
extinct nor free; **(b)** heritable variance, so differences compound; **(c)**
coupling to the function you want, so the gradient points at it. Here, the
information channel has (b) but not (a) or (c); the attention channel has (a)
approximately and neither (b) nor (c). Hand-tuning is what currently substitutes
for all three, which is exactly the trap in the question: the tuned parameters are
not helping evolution start, they are standing in for it.

The corollary is that scaffolding (subsidies, grace periods, juvenile discounts,
minimum-population floors) is the *last* lever, not the first. Scaffolding a
gradient that points the wrong way just gets you there faster.

## The trees (each domain's version is different)

* **TattleTots** — as above: initiation succeeds trivially and means nothing. The
  binding failure is currency design, not cold start.
* **Coral Key** — the observed side cannot evolve at all. The fleet is a
  hand-counted 15 legal / 4 gaming / 3 IUU set with no reproduction, mutation, or
  death (`fleet/behavior.py`), departing with fixed probability 0.15/epoch. So
  there is no arms race; detectability is a fixed target. That is *good* for
  initiation — a stationary target means a gradient exists from step 0 — and it
  bounds the claim: nothing observed here can be evidence about coevolution.
* **Scrapiron (fire)** — the initiation problem is *masked*, which is worse than
  failing. A4 appends an OPIR backstop to detections unconditionally
  (`architectures/a4_bma.py`), so domain metrics survive total agent extinction.
  Fire currently cannot tell a working ecology from a dead one.
* **Xylella (grain)** — the only working evolutionary loop in the whole family is
  on the **pests**: resistance frequency starts at 0.01, pesticide kills
  susceptibles, and behavioral escape appears after a species-specific generation
  counter (`environment/pest.py`). Near break-even mean, heritable variance,
  coupled to the trait under selection. This is the reference shape to copy — and
  note that the adversary evolving while the agents effectively do not is a
  measurement hazard, not a symmetry.
* **domain-runner** — no population model at all, by design; it contributes
  nothing to initiation either way.

## What to do, in order

1. **Measure initiation before engineering it.** A run must not be able to report
   success while being wrong roughly 99% of the time or while its yield is
   ungrounded. Precision above chance is achievable in some cohorts, so the
   question is now the gradient and its persistence, not whether any cohort can
   ever be right. Minimum: correct-report rate,
   per-capita attention solvency, and share of yield traceable to ground-truth
   streams, per run; for fire, agent-only detections with OPIR ablated. Without
   this, every later change is unfalsifiable.
2. **Couple the currencies.** Either fund the scarce currency from being right
   (attention income earned by verified-correct reports rather than
   trust×relevance on unverified signal vectors), or make information yield
   discountable by downstream verification so that compressing noise does not pay.
   The design invariant to hold: whatever gates reproduction must be capped by the
   same carrying capacity that gates survival.
3. **Price reproduction against function**, so precocity is not the dominant
   gradient — pay for offspring in the scarce currency.
4. **Only then** consider innate calibration (adaptive escalation thresholds as the
   initiation default, with fixed thresholds allowed to win later) and an annealed
   subsidy — each with an ablation that shows how much of the final performance it
   is carrying.

## What would falsify this

A run in which correct-report rate rises over generations without a change to
initial parameters, or a parent–child reproductive correlation reliably above ~0.2.
Either would show a functional gradient already exists and that the diagnosis above
is measuring the wrong thing. As of this measurement, precision above chance is
already achievable in some cohorts, but grounded yield remains below minimum in
20/20 runs and the parent–child correlation is near zero and nonsignificant.

## Provenance

Diagnostic design: three arms × 20 replicates — Arm B is confounded by
seed-derived identity as well as environment; Arm C holds the environment and
world identities fixed while genomes vary — plus a within-run selection
differential, a threshold-band mortality split, a parent–offspring correlation,
and per-currency energy attribution. The diagnostic is now reproducible. The
earlier figures shifted because the seed bug fixed in TattleTots #22 made seeded
runs reproducible; no engine mechanics were changed for this re-baseline, and no
engine code was modified to obtain these numbers.
