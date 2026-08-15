# Fixing the payoff path: verified-correctness income and merit rationing

Follows `docs/currency-coupling-diagnosis.md`, which located two serial breaks in the
chain `correctness -> trust -> attention income -> reserves -> reproduction`:
trust never converted into meaningful attention income, and once income was made
abundant the population cap rationed reproduction by agent creation order.

Two engine mechanisms address those two causes. Both are config-gated and
default-off, so the default engine is byte-identical in behaviour to before.

| Knob | Default | Effect when raised |
|---|---|---|
| `SimulationConfig.correct_report_attention_value` | `0.0` | Pays the per-user value term `v` in `income = alpha * v` for each verified-correct report, so attention income depends on being right rather than on looking relevant. |
| `SimulationConfig.reproduction_merit_ordering` | `False` | Orders eligible parents by the uncapped scarcer-currency sufficiency (`Agent.reproduction_sufficiency`) before `max_population` truncates the list, so a binding cap rations by reserves instead of creation order. |

`Agent.reproduction_limiting_factor` is unchanged in value; it now derives from the
new uncapped sufficiency helper, which is what makes merit ordering possible at all
(the capped limiting factor is exactly 1.0 for every solvent agent and cannot rank
them).

## How to reproduce

```bash
bash scripts/run_payoff_fix_ablation.sh                    # 5 arms x 2 policies x 5 seeds
uv run --no-sync --no-build python scripts/check_falsification_reliability.py
```

SparseSensor, 200 steps, seeds 42-46 (reliability check: 42-61), grounded input
fraction 0.67, initial population 20, cap 60 — identical across all cells, so any
difference is caused by the knobs alone.

## Result 1: the severed links are repaired, gradedly

Ordinary (evolved) arm, Pearson r over adults, mean of 5 seeds:

| Link | baseline | value only | merit only | value + merit |
|---|---|---|---|---|
| correctness -> attention income | +0.032 | +0.210 | +0.005 | +0.091 |
| correctness -> offspring | −0.013 | +0.085 | −0.023 | +0.044 |
| parent-child offspring r | +0.114 | +0.097 | +0.143 | +0.127 |

Dose response for `correct_report_attention_value` with merit ordering on
(ordinary arm, 5 seeds) — the link is monotone in the knob, which is the evidence
that the mechanism, not a seed, carries the effect:

| value | 0 | 2 | 8 | 32 | 128 |
|---|---|---|---|---|---|
| correctness -> attention income | +0.005 | +0.063 | +0.091 | +0.345 | +0.603 |
| correctness -> offspring | −0.023 | −0.010 | +0.044 | +0.195 | +0.439 |
| correct-report rate | 11.44% | 12.57% | 10.40% | 10.94% | 16.36% |
| population-capped step share | 10.7% | 7.9% | 8.2% | 13.2% | 34.3% |

Unlike the 20x attention-budget ablation in the previous diagnosis, this restores
correctness -> offspring *without* inflating everyone's income: at value 128 the
correctness -> offspring correlation reaches +0.44 where the budget ablation
stalled at +0.14, because merit ordering stops the cap from discarding the
surplus.

## Result 2: neither falsification clause clears, and a third break is why

Per-seed reliability over 20 seeds, ordinary arm, merit ordering on:

| dose | correct-report rate | clause 1: slope/generation (seeds rising) | clause 2: parent-child offspring r (seeds > 0.2) |
|---|---|---|---|
| 0 | 10.99% | −0.0008 (7/20) | +0.131 (2/20) |
| 32 | 10.84% | +0.0000 (12/20) | +0.071 (0/20) |
| 128 | 14.15% | +0.0011 (14/20) | −0.030 (0/20) |

Clause 1 turns from majority-falling to majority-rising as the knob rises, but
+0.001/generation in 14/20 seeds is not a reliable within-run rise. Clause 2 moves
the wrong way: paying hard for correctness *reduces* parent-child offspring
correlation.

The new metric explains both. Parent-child correlation of *precision* — whether a
good reporter has good children at all — is:

| arm | baseline | value only | merit only | value + merit |
|---|---|---|---|---|
| ordinary (evolved) | +0.021 | +0.041 | +0.044 | +0.041 |
| oracle invasion | +0.662 | +0.601 | +0.688 | +0.629 |

Among evolved agents correctness is essentially **not heritable** (r ≈ 0.02–0.13),
while the oracle lineage — correct by construction and therefore heritable by
construction — shows r ≈ 0.6–0.7 in the same runs with the same measurement. With
a 16% decoder ceiling and a 3% event base rate, an individual agent's precision is
dominated by sampling noise, so a correctness-priced currency pays luck. That is
why raising the dose raises correctness -> offspring (the payment works) while
lowering parent-child offspring correlation (what it pays for is not transmitted).

## What this means for the next lever

The payoff path is now three breaks, not two, and the first two are fixed:

1. correctness -> income — fixed by `correct_report_attention_value`, graded and monotone.
2. income -> differential offspring at the cap — fixed by `reproduction_merit_ordering`.
3. **correctness -> heritable correctness** — open. Selection has nothing to act on
   because evolved precision carries almost no between-lineage variance.

Discounting information yield by verification (the third proposed intervention)
shrinks the correctness-blind path, which sharpens 1 further; on this evidence it
cannot by itself clear either clause, because break 3 is upstream of payoff
entirely. The candidates for break 3 are the instrument's own ceiling (a 16%
decoder precision leaves little heritable spread to select on) and the genome's
lack of a trait that changes precision, both of which are measurable with the
existing ceiling harness before any further engine change.

Defaults are unchanged: `correct_report_attention_value=0.0` and
`reproduction_merit_ordering=False` reproduce the previous engine exactly, and the
tables above are diagnostics, not new defaults.
