# Rationing reproduction by verified correctness

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Arm: `ordinary`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Grounded input fraction (fixed): `0.67`
- Max population (fixed): `60`
- `correct_report_attention_value` (fixed across arms): `8`
- `false_alarm_break_even_precision` (fixed across arms): `0.2`
- `escalation_calibration_in_score_units` (fixed across arms): `True`
- `reproduction_merit_ordering` (fixed across arms): `True`

Every arm shares the same initial parameters; the arms differ only in `reproduction_correctness_weight`, the share of reproductive merit carried by rank in verified correctness rather than rank in reserve sufficiency. The `correctness_weight_0` arm is the reserves-only ordering every earlier lever was measured under.

| Quantity | `correctness_weight_0` | `correctness_weight_0.25` | `correctness_weight_0.5` | `correctness_weight_1` |
|---|---|---|---|---|
| Realized break-even precision | 20.00% | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0469 | 0.0470 | 0.0471 | 0.0452 |
| Attention value of a correct report | 0.1874 | 0.1880 | 0.1883 | 0.1807 |
| Correct-report rate | 11.41% | 15.04% | 11.83% | 12.88% |
| Reports per adult lifetime | 9.01 | 16.17 | 7.30 | 11.51 |
| Adult steps per agent | 13.55 | 23.10 | 11.72 | 16.29 |
| Silent-adult share | 25.60% | 17.49% | 25.13% | 15.17% |
| Attention income / agent-step | 0.0265 | 0.0292 | 0.0264 | 0.0271 |
| Fitness alignment b (correctness -> offspring) | +0.017 | +0.058 | +0.020 | +0.066 |
| Clause 1: correct-report slope / generation | -0.0001 | +0.0026 | +0.0010 | +0.0023 |
| Clause 2: parent-child offspring correlation | +0.076 | -0.030 | +0.083 | +0.054 |
| Parent-child precision correlation | +0.171 | +0.123 | +0.138 | +0.168 |
| Parent-child pairs | 750.4 | 508.8 | 813.6 | 630.6 |
| Adults scored | 585.6 | 408.6 | 626.6 | 499.2 |
| Final population | 59.8 | 59.8 | 56.2 | 59.4 |
