# Population scale against the falsification clauses

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Arm: `ordinary`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Grounded input fraction (fixed): `0.67`
- Max population (fixed): `60`
- `correct_report_attention_value` (fixed across arms): `8`
- `false_alarm_break_even_precision` (fixed across arms): `0.2`
- `escalation_calibration_in_score_units` (fixed across arms): `True`
- Reference cap the earlier levers were measured at: `60`

Every arm shares the same initial parameters except the population cap and the founding population, which is a fixed share of the cap. The `Max population` line above is the sweep default; each arm's own cap is its column label. The `_per_capita` arms additionally scale the users' attention budget with the cap, so per-capita solvency stays at its reference value instead of falling as the population grows.

| Quantity | `cap_60` | `cap_125` | `cap_250` | `cap_500` | `cap_125_per_capita` | `cap_250_per_capita` | `cap_500_per_capita` |
|---|---|---|---|---|---|---|---|
| Realized break-even precision | 20.00% | 20.00% | 20.00% | 20.00% | 20.00% | 20.00% | 20.00% |
| Attention charged per false alarm | 0.0469 | 0.0289 | 0.0168 | 0.0157 | 0.0443 | 0.0456 | 0.0452 |
| Attention value of a correct report | 0.1874 | 0.1156 | 0.0671 | 0.0629 | 0.1771 | 0.1822 | 0.1810 |
| Correct-report rate | 11.41% | 9.62% | 9.59% | 9.85% | 11.04% | 11.50% | 12.23% |
| Reports per adult lifetime | 9.01 | 3.81 | 2.89 | 2.82 | 8.97 | 11.39 | 13.15 |
| Adult steps per agent | 13.55 | 7.39 | 5.85 | 5.72 | 13.31 | 15.78 | 17.32 |
| Silent-adult share | 25.60% | 31.96% | 35.54% | 35.18% | 18.44% | 19.14% | 17.56% |
| Attention income / agent-step | 0.0265 | 0.0157 | 0.0089 | 0.0084 | 0.0248 | 0.0258 | 0.0262 |
| Fitness alignment b (correctness -> offspring) | +0.017 | +0.047 | +0.055 | +0.033 | +0.025 | +0.023 | +0.044 |
| Clause 1: correct-report slope / generation | -0.0001 | +0.0007 | -0.0018 | -0.0002 | +0.0002 | -0.0004 | +0.0004 |
| Clause 2: parent-child offspring correlation | +0.076 | +0.088 | +0.118 | +0.102 | +0.040 | +0.043 | +0.015 |
| Parent-child precision correlation | +0.171 | +0.162 | +0.111 | +0.129 | +0.168 | +0.165 | +0.172 |
| Parent-child pairs | 750.4 | 1636.6 | 2815.8 | 2946.2 | 1536.2 | 2653.8 | 4958.2 |
| Adults scored | 585.6 | 1250.8 | 2118.6 | 2317.4 | 1189.0 | 2075.6 | 3867.4 |
| Final population | 59.8 | 112.2 | 211.0 | 200.6 | 124.8 | 249.8 | 499.6 |
