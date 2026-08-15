> **Superseded provenance notice**
>
> This document was produced by the harness at `/home/ubuntu/initiation-diag`,
> which was never committed and no longer exists. Its results are therefore
> not reproducible from this repository. The `gaussian_shift` scenario has no
> modeled instruments and a vacuous localization null: there is a single event
> location, so the static prior is 100%. As recorded by the sensor-check
> exemption in TattleTots #44, this scenario cannot support a claim about
> whether competence is expressible in genome space.
>
> The headline conclusion here is contradicted by measurement on a real
> instrument: Coral's hand-designed reporter reached 26.2% all-designed
> precision and 32.8% invasion precision against a 14.84% static-prior null
> (20 seeds, cap 48; Coral `docs/designed_reporter_measurement.md`).
>
> The reward-magnitude observations still concern the engine's economy, not
> the `gaussian_shift` instrument. The cost-structure observations likewise
> concern the engine's economy, not that instrument. Both were measured on a
> superseded engine, so their magnitudes are indicative only.

# Currency reward ceiling and oracle test

Date: 2025-02-14

This is a measurement-only harness. No TattleTots repository files were changed.
The harness is `/home/ubuntu/initiation-diag/ceiling_harness.py`; raw JSON outputs are in the same directory.

The two arms are configurations of the now-merged engine:

- **Legacy/coupling-off:** `reproduction_coupling_strength=0.0`, `grounding_quality_strength=0.0`, requirement scales `1.0`.
- **Coupled/default:** default co-limitation and grounding settings (`reproduction_coupling_strength=1.0`, information and attention scales `1.0`, `grounding_quality_strength=0.5`).

The branch checkout used for the coupled arm was the pre-merge PR #25 snapshot. It is code-identical to the subsequently merged `main`, per the handoff; the arms should therefore be read as configuration arms, not repository arms.

All runs use GaussianShift, 200 steps, and the existing diagnostic parameters:

```text
initial_population=20
max_population=60
mutation_rate=0.1
recombination_probability=0.3
false_alarm_penalty=0.4
subsidy_rate=0.1
```

Seeds used were 42, 7, and 99 for invasion runs. The monoculture and payment-path headline runs use seed 42.

## Verdict first

This is a reality problem in two specific places, not a viability problem:
correctness reaches the individual at too small a magnitude, and useful
competence is not expressible in the tested genome space. A perfect reporter
can sustain a monoculture in both arms, so the ecology can carry competence once
it is supplied.

However, the gradient is weak and stochastic:

- A correct report's marginal same-step attention-income gain is approximately `0.0009–0.0026` energy units in the observed runs.
- Typical per-step compression income is approximately `1.27` in the legacy arm and `0.56` in the coupled arm.
- The measured mean cross-agent per-step SD of compression income is `3.33` and `0.77`, respectively; using the earlier diagnostic yardstick of approximately `12` makes the reward-to-noise ratio smaller still.
- The genome grid found no precision ceiling above the empirical chance baseline (approximately `2.5%`) in either arm.
- Oracle invasion is strongly seed-dependent: one legacy seed and one coupled seed lose the oracle lineage, while the other seeds show growth.

Thus the evidence supports two separate reality failures: the trickle-down
reward is too weak to detect, and ordinary genomes cannot express a useful
reporter under this scenario. The oracle invasion spread is consistent with
drift: an oracle lineage can win or lose depending on stochastic survival and
reproduction, not because its correctness produces a reliably detectable
advantage. Raising correctness income is therefore premature until competence
is expressible.

## 1. Payment-path arithmetic

The code path is:

1. `World._maybe_escalate()` creates a `Report`.
2. `trust.verify_reports()` sets:

   ```python
   report.verified = True
   report.correct = report.location in active_locations
   ```

3. For a correct report, the target user updates trust by:

   ```python
   user.update_trust(
       report.agent_id,
       TrustOutcome.CORRECT_ALARM,
       deltas=TrustUpdateDeltas(pos=config.trust_delta_pos),
   )
   ```

   The default `trust_delta_pos` is `0.05`.

4. Attention allocation then computes, for each user:

   ```python
   score_i = trust_i * relevance_i
   weight_i = score_i / sum(score_j)
   allocation_i = attention_budget * weight_i
   ```

5. Agent attention income is:

   ```python
   income = sum(allocation_i * verified_value)
   ```

   with `verified_value=1.0`.

6. The energy update is:

   ```python
   info_delta = -compute_cost + information_yield + downstream_subsidy
   attention_delta = attention_income - juvenile_maintenance - false_alarm_count * false_alarm_penalty
   ```

   A correct report has no false-alarm penalty. There is no separate per-report agent energy debit in the engine. The Gaussian scenario's external accounting additionally charges `0.1` surveillance cost per report and `1.0` response cost per correct report.

The harness recomputed attention allocation in the same state with the `+0.05` trust update removed. The difference is the measured marginal attention income. It is exact away from trust saturation at 1.0; saturation cases are retained and identified as a conservative same-state counterfactual rather than discarded.

### Payment measurements

| Run | Correct-report rows | Mean marginal attention income | Median | Maximum | Mean compression income | Mean cross-agent SD of compression income |
|---|---:|---:|---:|---:|---:|---:|
| Legacy ordinary population, seed 42 | 0 | 0 | 0 | 0 | 1.2746 | 3.3349 |
| Coupled ordinary population, seed 42 | 0 | 0 | 0 | 0 | 0.5623 | 0.7730 |
| Legacy oracle monoculture, seed 42 | 269 | 0.000891 | 0 | 0.011814 | 0.8579 | 2.7365* |
| Coupled oracle monoculture, seed 42 | 286 | 0.001111 | 0 | 0.011099 | 0.5675 | 1.6387* |

`*` The oracle rows also include inactive/event-free steps and are reported as an overall income SD. The ordinary-population cross-agent SD is the cleaner noise yardstick. Against the earlier stated information-income SD of approximately 12, the mean correct-report reward is only about `0.007–0.009%` of that yardstick.

In the invasion runs, the marginal reward was:

| Arm | Seed | Correct-report rows | Mean marginal attention income | Mean compression income | Correct reports / verified reports |
|---|---:|---:|---:|---:|---:|
| Legacy | 42 | 10 | 0.001856 | 0.8462 | 10 / 10 |
| Legacy | 7 | 180 | 0.001593 | 1.3423 | 180 / 180 |
| Legacy | 99 | 116 | 0.001617 | 1.4793 | 116 / 116 |
| Coupled | 42 | 0 | 0 | 0.5663 | 0 / 0; oracle extinct before event window |
| Coupled | 7 | 64 | 0.002592 | 0.4698 | 64 / 64 |
| Coupled | 99 | 285 | 0.001170 | 0.3589 | 285 / 285 |

The ordinary runs had no correct reports at all in seed 42, so their direct reward rows are necessarily empty. This is itself evidence of the cold-start problem, not evidence that the payment path is broken.

## 2. Feasibility ceiling

I ran a 432-point genome grid per arm and seed over:

- all four sensing strategies;
- all four compression types;
- working dimensions `8`, `20`, and `30`;
- escalation thresholds `0.05`, `0.3`, and `0.7`;
- spatial strategies global, peak, and weighted ROI.

Each candidate was run as a two-agent monoculture for 200 steps with reproduction disabled by `max_population=2`. The two copies share the candidate genome; the event stream and ordinary verification path remain unchanged.

### Best grid results

| Arm | Seed | Best precision | Correct / reports | Best configuration |
|---|---:|---:|---:|---|
| Legacy | 42 | 2.525% | 10 / 396 | subspace sample, wavelet, dim 8, threshold 0.05, weighted ROI |
| Coupled | 42 | 2.525% | 10 / 396 | subspace sample, PCA, dim 20, threshold 0.05, weighted ROI |
| Legacy | 7 | 2.273% | 9 / 396 | concat, wavelet, dim 20, threshold 0.05, peak |
| Coupled | 7 | 2.525% | 10 / 396 | concat, AR1, dim 30, threshold 0.05, weighted ROI |

The scenario event prevalence is `11 / 200 = 5.5%` in the active-location window by step count. The engine's chance-precision baseline is computed from mean active-location count divided by observed location support; in these high-report candidates the resulting null is approximately `2.5%`. The best candidates therefore do not establish a useful precision advantage; they sit at the empirical null level. No grid candidate demonstrated a robust above-chance precision ceiling.

This is a grid ceiling, not a mathematical proof over every possible genome. It is nevertheless strong evidence that the ordinary sensing/reporting mechanics do not expose a high-precision strategy in this scenario.

## 3. Oracle monoculture

The harness-only oracle overrides only the decision to escalate. It reports exactly the active event location when an event is active and otherwise remains silent. It still uses:

- ordinary stream attachment;
- ordinary sensing and compression;
- ordinary information compute cost and yield;
- ordinary attention allocation;
- ordinary trust updates;
- ordinary reproduction and mutation;
- ordinary death checks and maintenance;
- ordinary false-alarm handling;
- ordinary external Gaussian surveillance/response costs.

The oracle report is verified before attention allocation. This was explicitly checked: every oracle report in the tables below was verified and correct. There was no silent unverified-report path.

| Arm | Seed | Final population | Births | Deaths | Oracle reports | Verified | Correct | Precision | Solvent fraction | Raw grounded share | Effective grounded share | Degeneracy reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Legacy | 42 | 60 | 339 | 102 | 269 | 269 | 269 | 100% | 36.5% | 22.6% | 22.6% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Coupled | 42 | 59 | 401 | 151 | 286 | 286 | 286 | 100% | 37.4% | 17.7% | 28.0% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |

Both oracle monocultures remain near the population ceiling for the full run. This rules out “a perfect reporter cannot sustain itself” as the explanation.

## 4. Oracle invasion

The invasion starts with two oracle agents and eighteen ordinary agents. Oracle status is inherited by any child with an oracle parent, while all reproduction continues through the ordinary reproduction machinery.

### Summary by seed

| Arm | Seed | Final population | Final oracle share | Maximum oracle share | First step >50% | Births | Deaths | Oracle precision | Solvent fraction | Raw grounded | Effective grounded | Degeneracy reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Legacy | 42 | 59 | 3.4% | 28.8% | — | 340 | 103 | 100% (10/10) | 36.4% | 23.1% | 23.1% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Legacy | 7 | 60 | 60.0% | 60.0% | 150 | 359 | 102 | 100% (180/180) | 34.7% | 19.8% | 19.8% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Legacy | 99 | 58 | 53.4% | 53.4% | 197 | 547 | 141 | 100% (116/116) | 34.2% | 21.9% | 21.9% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Coupled | 42 | 59 | 0.0% | 9.7% | — | 402 | 151 | no oracle reports; lineage extinct before event | 37.2% | 17.9% | 28.2% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Coupled | 7 | 60 | 25.0% | 25.4% | — | 390 | 161 | 100% (64/64) | 36.6% | 17.8% | 28.5% | insufficient location support; grounded yield below minimum; attention insolvency with capacity overshoot |
| Coupled | 99 | 60 | 95.0% | 95.0% | 25 | 214 | 80 | 100% (285/285) | 46.1% | 18.8% | 29.7% | insufficient location support; grounded yield below minimum |

### Sampled oracle-share trajectories

The raw JSON files contain all 200 points. The following samples are every 25 steps, including step 0 and the final step.

```text
Legacy, seed 42: 12.1%, 25.5%, 19.0%, 15.0%, 11.9%, 5.0%, 6.7%, 3.4%
Legacy, seed 7:  12.9%, 10.3%, 22.0%, 41.7%, 41.7%, 41.7%, 51.7%, 58.3%
Legacy, seed 99:  9.4%, 19.0%, 22.4%, 25.0%, 34.5%, 42.4%, 43.9%, 44.8%

Coupled, seed 42:  8.3%, 3.4%, 1.7%, 1.7%, 3.4%, 20.0%, 0.0%, 0.0%
Coupled, seed 7:  12.5%, 5.2%, 6.9%, 10.0%, 13.3%, 20.0%, 20.0%, 25.0%
Coupled, seed 99: 12.0%, 51.7%, 75.0%, 83.3%, 88.3%, 90.0%, 91.7%, 95.0%
```

The coupled seed-42 lineage disappears before the event window, so it produces no oracle report. This is an important distinction from an unverified oracle: its reports are absent because its lineage is absent, not because its reports were ignored. In every run where an oracle reported, verification was `100%`.

## Reproducibility

A fresh separate-process rerun of the seed-7 invasion runs was byte-identical:

```text
Legacy seed 7:
79e0ad779746d0b142c5e9bcac25040b0909e01de0ef9b40540ee491c3fe981a
79e0ad779746d0b142c5e9bcac25040b0909e01de0ef9b40540ee491c3fe981a
cmp=0

Coupled seed 7:
ffce1c4828e9fc3eb1d2a6a29ea2a0236d2550e18bb8fe9daa9ae56e12295136
ffce1c4828e9fc3eb1d2a6a29ea2a0236d2550e18bb8fe9daa9ae56e12295136
cmp=0

Legacy seed 99:
c73d5cce458b634f66f2e4f56e6c2524c5528e90225156ff2c4977a644b48ce8
c73d5cce458b634f66f2e4f56e6c2524c5528e90225156ff2c4977a644b48ce8
cmp=0

Coupled seed 99:
7072080bb01ce56ddd46236f77f6dfb2742074ab020f34045c104336a5582726
7072080bb01ce56ddd46236f77f6dfb2742074ab020f34045c104336a5582726
cmp=0
```

The deterministic engine and harness therefore reproduce representative reported per-seed trajectories (seeds 7 and 99 in both configuration arms).

## Files

Main raw results:

- `payment-legacy-42.json`
- `payment-coupled-42.json`
- `oracle-main-mono-42.json` / `oracle-coupled-mono-42.json` (pre-merge snapshot names; these correspond to the legacy/coupled config arms)
- `oracle-main-invasion-{42,7,99}.json`
- `oracle-coupled-invasion-{42,7,99}.json`
- `ceiling-main-{42,7}.json`
- `ceiling-coupled-{42,7}.json`

The `main`/`coupled` filename distinction is retained only as an artifact of the two pre-merge checkout paths. The substantive comparison is configuration-based as described above.
