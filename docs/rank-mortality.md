# Rank-coupled mortality: does a low-ranked adult die while waiting?

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Arm: `ordinary`
- Steps per run: `600`
- Seeds: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61`
- Grounded input fraction (fixed): `0.67`
- Max population (fixed): `60`
- `correct_report_attention_value` (fixed across arms): `8`
- `false_alarm_break_even_precision` (fixed across arms): `0.2`
- `escalation_calibration_in_score_units` (fixed across arms): `True`
- `reproduction_merit_ordering` (fixed across arms): `True`
- `reproduction_correctness_weight` (fixed across arms): `1`
- `reproduction_recruitment_share` (fixed across arms): `1`

Every arm shares the same initial parameters; the arms differ only in the scale applied to each user's attention budget, the currency whose insolvency causes essentially every death. The `budget_scale_1` arm is the environment every earlier lever was measured under.

| Quantity | `budget_scale_1` | `budget_scale_0.5` | `budget_scale_0.25` |
|---|---|---|---|
| Realized break-even precision | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0423 | 0.0246 | 0.0145 |
| Attention value of a correct report | 0.1694 | 0.0984 | 0.0579 |
| Correct-report rate | 14.32% | 11.72% | 8.51% |
| Reports per adult lifetime | 37.26 | 16.78 | 6.13 |
| Adult steps per agent | 51.94 | 23.66 | 10.16 |
| Silent-adult share | 14.40% | 18.53% | 21.98% |
| Attention income / agent-step | 0.0265 | 0.0143 | 0.0080 |
| Fitness alignment b (correctness -> offspring) | -0.040 | +0.022 | +0.090 |
| Reproductive excess (eligible / slot) | 31.67 | 11.76 | 4.13 |
| Slot-limited step share | 92.50% | 75.06% | 56.55% |
| Opportunity for selection I | 0.875 | 0.538 | 0.300 |
| Mean offspring per adult | 1.27 | 1.29 | 1.31 |
| Died before the run ended (adults) | 90.04% | 94.46% | 97.18% |
| Childless adult share | 24.97% | 15.94% | 8.31% |
| Rank persistence (early vs late life) | +0.667 | +0.634 | +0.588 |
| Rank -> adult lifespan | +0.054 | +0.090 | +0.233 |
| Rank -> offspring | +0.359 | +0.291 | +0.140 |
| Lifespan -> offspring | -0.116 | -0.128 | -0.076 |
| Clause 1: correct-report slope / generation | +0.0034 | +0.0007 | -0.0003 |
| Clause 2: parent-child offspring correlation | -0.043 | +0.022 | +0.053 |
| Parent-child precision correlation | +0.152 | +0.165 | +0.129 |
| Parent-child pairs | 816.3 | 1409.9 | 2059.4 |
| Adults scored | 641.5 | 1086.8 | 1568.0 |
| Final population | 60.0 | 59.9 | 58.1 |
