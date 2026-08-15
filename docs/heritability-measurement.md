# Break 3: correctness is heritable, but a single agent's correctness is unmeasurable

`docs/payoff-fix-measurement.md` closed the first two payoff breaks and exposed a
third: paying for verified correctness raises correctness -> offspring monotonically,
yet parent-child correlation of *precision* among evolved agents is only ~0.02-0.13
against ~0.6-0.7 for the heritable-by-construction oracle lineage in the same runs.
Two candidate causes were named there: the instrument's own ceiling, or a genome with
no trait that changes precision.

Neither is the answer. The genome has strong leverage over precision; what is missing
is the per-individual sample size needed to see it.

## Reproduction

```bash
uv run --no-sync --no-build python scripts/measure_correctness_heritability.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 --clone-genomes 10 --clone-replicates 42 43 44
```

SparseSensor, 200 steps, `grounded_input_fraction=0.67`, initial population 20, cap 60,
engine defaults otherwise (both payoff knobs off). The script prints only; it writes no
artifacts.

## 1. Genomic leverage is large

Clone monocultures — every founder seeded with one genome, `mutation_rate=0` and
`recombination_probability=0` so the whole population stays genetically identical —
replicated across three environment seeds:

| metric | value |
|---|---|
| genomes x replicates | 10 x 3 |
| mean clone-run precision | 17.03% |
| between-genome variance | 0.0145 |
| within-genome (replicate) variance | 0.0085 |
| **intraclass correlation** | **0.630** |
| genome means, min -> max | 0.0% -> 34.9% |
| reports per run | 80.6 |

Genotype explains ~63% of the variance in run-level correct-report rate, and the
spread between genomes is the full width of the instrument: a bad genome reports at 0%
correct, the best at 34.9% — above the 16% decoder precision and far above the 3.00%
static-prior null. There is real, selectable variation in the genome.

## 2. A single agent never emits enough reports to reveal it

Same runs, evolved (ordinary) arm, 6350 adults over 10 seeds:

| metric | value |
|---|---|
| mean adult steps per agent | 6.76 |
| mean reports per adult lifetime | 0.457 |
| median reports per adult lifetime | 0 |
| max reports per adult lifetime | 9 |
| share of adults with >= 5 reports | 0.30% |

An agent's precision is therefore a Bernoulli estimate on ~0-1 trials. Conditioning
the parent-child precision correlation on report count cannot even be evaluated: at
>= 5 reports for both parent and child there is **1 pair** in 10 seeds, versus 822 at
>= 1 report.

The between-agent spread in precision is accordingly pure noise:

| metric | value |
|---|---|
| mean precision | 11.7% |
| observed between-agent variance | 0.0832 |
| binomial noise floor at each agent's report count | 0.0857 |
| **excess variance ratio** | **0.97** |

A ratio of 1.0 means every bit of apparent difference between agents is sampling luck.
There is no measured individual differential for selection to act on.

## 3. The observed weak heritability is exactly what the sample size predicts

With genomic variance 0.0145, pooled precision 0.117 and 0.457 reports per agent, a
per-agent precision estimate carries binomial error variance 0.226, so a heritable
trait measures attenuated by `var_g / (var_g + p(1-p)/n)`:

| quantity | value |
|---|---|
| attenuation factor | 0.060 |
| predicted parent-child precision r | **+0.038** |
| observed parent-child precision r (evolved) | −0.013 .. +0.124 |
| reports per agent needed to halve the attenuation | 7.2 |

The prediction lands inside the observed range. Break 3 needs no explanation beyond
sample size: correctness *is* heritable, and the reason selection does not see it is
that an agent issues ~0.46 reports before it dies, when ~7 are needed for its own
precision to carry half its genome's signal.

## Where that leaves the levers

Break 3 is not the instrument (the best clone genome clears the decoder precision) and
not an absent trait (ICC 0.63). It is *reporting opportunity per lifetime*, the product
of two things:

- adult lifespan ~6.8 steps, and
- ~0.068 reports per adult step.

Even an immortal agent at the current reporting rate would need ~100 adult steps to
reach 7 reports. Selection cannot act on a phenotype nobody expresses often enough to
be scored on.

Both throttles have since been measured in `docs/reporting-opportunity-measurement.md`:
evidence arrival is *not* limiting (93% of adult steps carry grounded yield), the
escalation threshold sits ~0.56 above the median anomaly so 79.5% of evolved adults
never report at all, 97% of deaths are attention insolvency, and at ~12% precision
escalating makes an agent's attention balance *worse* (corr(escalation rate, attention
drift) = −0.17) while for the oracle lineage the same mechanism runs the other way
(+0.33, and reporting nearly triples adult lifespan). Silence is the evolved optimum.

Neither falsification clause clears here, and nothing in this document changes engine
behavior: it adds a measurement script and no defaults.
