# Which throttle limits reporting opportunity: silence, not evidence

`docs/heritability-measurement.md` showed that correctness is heritable (clone
intraclass correlation 0.63) but unmeasurable per individual, because an adult issues
~0.46 reports in a ~6.8-step adult life when ~7.2 reports are needed for its own
precision to carry half its genome's signal. Lifetime reports are lifespan x reporting
rate, so this measurement asks which of the two the engine throttles, and why.

## Reproduction

```bash
uv run --no-sync --no-build python scripts/measure_reporting_opportunity.py \
  --seeds 42 43 44 45 46 47 48 49 50 51
```

SparseSensor, 200 steps, `grounded_input_fraction=0.67`, initial population 20, cap 60,
engine defaults otherwise (both payoff knobs off). `ordinary` is the evolved arm;
`oracle_invasion` seeds one hand-designed correct reporter lineage into the same world.
The script prints only; it writes no artifacts.

## Evidence arrival is not the throttle

The grounding fix did its job. Adult steps carrying grounded (raw-domain) yield:

| arm | share of adult steps with grounded yield |
|---|---|
| ordinary | 93.0% |
| oracle invasion | 94.2% |

The 1.41% evidence-starvation figure from the Coral measurement is gone. Agents have
the evidence on almost every step they are alive.

## The escalation threshold is the throttle

| per-adult-step funnel | ordinary | oracle invasion |
|---|---|---|
| grounded yield present | 93.0% | 94.2% |
| anomaly >= effective threshold | **3.67%** | 21.1% |
| escalated | 3.67% | 77.9% |
| reports issued per adult step | 0.068 | 0.829 |
| median (anomaly − threshold) | **−0.562** | −0.405 |
| p90 (anomaly − threshold) | −0.198 | +0.209 |

In the evolved arm the normalized anomaly sits roughly half a threshold-unit *below*
the firing threshold at the median, and still below it at the 90th percentile: the
threshold is not marginally too high, it is above nearly the whole anomaly
distribution. 79.5% of evolved adults never escalate once in their lives. The oracle
policy, which decides for itself rather than waiting for the threshold, escalates on
78% of its adult steps in the same worlds.

## Lifespan is capped by attention, and reporting does not buy attention back

| arm | mean adult steps | juvenile share of life | deaths | by attention | by information |
|---|---|---|---|---|---|
| ordinary | 6.76 | 47.9% | 91.7% of agents | **97.1%** | 2.9% |
| oracle invasion | 12.34 | 31.4% | 88.8% of agents | 96.5% | 3.5% |

Information is not scarce — it accumulates at +0.63/step and only 3.4% of adults have
negative information drift. Attention is the killer: it drifts at −0.040/step for
89.5% of evolved adults, and essentially every death is attention insolvency.

Attention income is only paid on reports, so the two throttles are one loop — and in
the evolved arm the loop runs the wrong way:

| arm | silent adults | adult steps, silent | adult steps, reporting | corr(escalation rate, lifespan) | corr(escalation rate, attention drift) |
|---|---|---|---|---|---|
| ordinary | 79.5% | 6.79 | 8.92 | **−0.043** | **−0.174** |
| oracle invasion | 28.5% | 5.99 | **16.05** | **+0.433** | +0.327 |

For an evolved agent reporting at ~12% precision, escalating makes the attention
balance *worse*: the false-alarm penalty (0.4) dwarfs the attention income a report
earns (~0.02), so speaking up shortens life and the population is selected toward
silence. For the oracle lineage, whose reports are almost always right, the identical
mechanism reverses sign and reporting nearly triples adult lifespan. The engine's
payoff loop is intact; at achievable precision it is simply priced against reporting.

## What it would take

| requirement | value |
|---|---|
| reports per lifetime needed (halve the heritability attenuation) | 7.2 |
| current, ordinary arm | 0.46 |
| adult steps needed at the current escalation rate | 106 (16x current lifespan) |
| reports per adult step needed at the current lifespan | 1.07 (16x current rate) |
| oracle invasion, same worlds | 10.2 reports per lifetime |

Neither throttle can deliver 7.2 reports alone at a plausible factor, and the oracle
arm already clears it — which locates the fix in the escalation/pricing loop rather
than in lifespan extension.

## Where that leaves the levers

Break 3 is not evidence starvation, not the instrument, and not a missing genomic
trait. It is that an evolved agent's *best move is silence*: the escalation threshold
sits above its anomaly distribution, and if it fires anyway the false-alarm penalty
costs more attention than a correct report earns. Both are engine pricing, not
scaffolding:

- the false-alarm penalty vs. attention income ratio sets a break-even precision far
  above the decoder ceiling (`docs/currency-coupling-diagnosis.md` measured 80%), so
  the sign of the return on reporting is negative for every reachable genome; and
- adaptive threshold calibration is supposed to bring the threshold to the anomaly
  distribution, but the measured median gap of −0.56 says it does not.

Adding a subsidy or a floor would be scaffolding and is excluded. Repricing the
false-alarm penalty against the instrument's achievable precision, or calibrating the
escalation threshold to the agent's own anomaly distribution, are candidate changes to
measure next — each with the same falsification test and each ablated.

`docs/domain-richness-requirement.md` converts these two throttles into the analytic
requirement they should be judged against, and finds that the 7.2 reports/lifetime bar
quoted above is an *observer's* bar: selection itself needs ~0.9 reports per agent at
perfect fitness alignment, or ~26 at the alignment measured so far.

Nothing in this document changes engine behavior: it adds a measurement script and no
defaults.
