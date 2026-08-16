# Repricing false alarms against reachable precision

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Arm: `ordinary`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Grounded input fraction (fixed): `0.67`
- Max population (fixed): `60`
- `correct_report_attention_value` (fixed across arms): `8`

Every arm shares the same initial parameters; only `false_alarm_break_even_precision` differs.

| Quantity | `flat_penalty` | `break_even_0.4` | `break_even_0.2` | `break_even_0.1` | `break_even_0.05` |
|---|---|---|---|---|---|
| Realized break-even precision | 63.63% | 40.00% | 20.00% | 10.00% | 5.00% |
| Attention charged per false alarm | 0.2711 | 0.1305 | 0.0480 | 0.0224 | 0.0102 |
| Attention value of a correct report | 0.2108 | 0.1958 | 0.1920 | 0.2012 | 0.1929 |
| Correct-report rate | 10.40% | 10.48% | 12.27% | 11.24% | 11.42% |
| Reports per adult lifetime | 0.45 | 4.11 | 3.52 | 3.39 | 3.13 |
| Adult steps per agent | 7.03 | 12.92 | 12.24 | 13.24 | 12.27 |
| Silent-adult share | 69.11% | 50.86% | 49.07% | 50.09% | 52.39% |
| Attention income / agent-step | 0.0268 | 0.0259 | 0.0255 | 0.0267 | 0.0253 |
| Fitness alignment b (correctness -> offspring) | +0.044 | +0.049 | +0.040 | +0.039 | +0.060 |
| Clause 1: correct-report slope / generation | -0.0002 | -0.0018 | -0.0008 | -0.0013 | -0.0006 |
| Clause 2: parent-child offspring correlation | +0.127 | +0.052 | +0.065 | +0.043 | +0.085 |
| Parent-child precision correlation | +0.041 | +0.194 | +0.188 | +0.086 | +0.045 |
| Parent-child pairs | 863.8 | 709.2 | 766.4 | 696.0 | 734.6 |
