# Reproductive excess and the opportunity for selection

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

Every arm shares the same initial parameters; the arms differ only in `reproduction_recruitment_share`, the share of the step's eligible parents that may recruit an offspring. The `recruitment_share_1` arm is the unlimited recruitment every earlier lever was measured under, where the population cap is the only limit.

| Quantity | `recruitment_share_1` | `recruitment_share_0.5` | `recruitment_share_0.25` | `recruitment_share_0.1` |
|---|---|---|---|---|
| Realized break-even precision | 20.00% | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0423 | 0.0421 | 0.0428 | 0.0440 |
| Attention value of a correct report | 0.1694 | 0.1682 | 0.1711 | 0.1760 |
| Correct-report rate | 14.32% | 13.95% | 14.20% | 14.40% |
| Reports per adult lifetime | 37.26 | 38.65 | 36.14 | 37.25 |
| Adult steps per agent | 51.94 | 54.09 | 50.55 | 54.95 |
| Silent-adult share | 14.40% | 14.02% | 12.84% | 4.63% |
| Attention income / agent-step | 0.0265 | 0.0263 | 0.0268 | 0.0281 |
| Fitness alignment b (correctness -> offspring) | -0.040 | -0.054 | -0.019 | -0.024 |
| Reproductive excess (eligible / slot) | 31.67 | 50.33 | 64.08 | 76.99 |
| Slot-limited step share | 92.50% | 100.00% | 100.00% | 100.00% |
| Opportunity for selection I | 0.875 | 0.796 | 0.874 | 1.037 |
| Mean offspring per adult | 1.27 | 1.26 | 1.28 | 1.27 |
| Clause 1: correct-report slope / generation | +0.0034 | +0.0030 | +0.0034 | +0.0033 |
| Clause 2: parent-child offspring correlation | -0.043 | -0.038 | -0.023 | -0.025 |
| Parent-child precision correlation | +0.152 | +0.156 | +0.134 | +0.146 |
| Parent-child pairs | 816.3 | 818.4 | 897.8 | 742.9 |
| Adults scored | 641.5 | 643.8 | 699.9 | 582.2 |
| Final population | 60.0 | 60.0 | 59.9 | 60.0 |
