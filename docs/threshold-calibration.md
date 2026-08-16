# Calibrating escalation thresholds to the compared distribution

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Arm: `ordinary`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Grounded input fraction (fixed): `0.67`
- Max population (fixed): `60`
- `correct_report_attention_value` (fixed across arms): `8`
- `false_alarm_break_even_precision` (fixed across arms): `0.2`

Every arm shares the same initial parameters; the arms differ in whether adaptive escalation thresholds are calibrated in the score units they are compared against, and in the starting range of the `escalation_threshold` trait.

| Quantity | `raw_units_control` | `score_units` | `score_units_start_0.1_0.5` | `score_units_start_0.05_0.3` |
|---|---|---|---|---|
| Realized break-even precision | 20.00% | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0480 | 0.0473 | 0.0473 | 0.0469 |
| Attention value of a correct report | 0.1920 | 0.1892 | 0.1893 | 0.1874 |
| Correct-report rate | 12.27% | 9.89% | 12.13% | 11.41% |
| Reports per adult lifetime | 3.52 | 1.94 | 8.34 | 9.01 |
| Adult steps per agent | 12.24 | 13.58 | 16.21 | 13.55 |
| Silent-adult share | 49.07% | 51.66% | 28.88% | 25.60% |
| Attention income / agent-step | 0.0255 | 0.0247 | 0.0272 | 0.0265 |
| Fitness alignment b (correctness -> offspring) | +0.040 | +0.070 | +0.027 | +0.017 |
| Clause 1: correct-report slope / generation | -0.0008 | -0.0011 | +0.0001 | -0.0001 |
| Clause 2: parent-child offspring correlation | +0.065 | +0.034 | +0.028 | +0.076 |
| Parent-child precision correlation | +0.188 | +0.213 | +0.133 | +0.171 |
| Parent-child pairs | 766.4 | 731.0 | 656.6 | 750.4 |
