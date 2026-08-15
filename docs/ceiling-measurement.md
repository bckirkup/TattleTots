# Ceiling measurement on a modeled instrument

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Static-prior null: **3.00%**
- Uniform (chance) null: **4.17%**
- Evidence inferability: **24.00%**
- Instrument valid: **True**

| Cell | Correct-report rate | Drift (2nd half − 1st) | Attention solvency | Grounded-yield share | Parent–child repro corr | Reports |
|---|---:|---:|---:|---:|---:|---:|
| `ordinary|fraction=0|multiplier=1` | 3.09% | -2.23% | 28.20% | 4.61% | 0.092 | 1356 |
| `ordinary|fraction=0.34|multiplier=1` | 12.80% | +0.87% | 40.62% | 52.97% | 0.114 | 1479 |
| `ordinary|fraction=0.67|multiplier=1` | 12.80% | +0.87% | 40.62% | 52.97% | 0.114 | 1479 |
| `ordinary|fraction=1|multiplier=1` | 12.80% | +0.87% | 40.62% | 52.97% | 0.114 | 1479 |
| `oracle_monoculture|fraction=0|multiplier=1` | 100.00% | +0.00% | 25.78% | 0.45% | 0.041 | 37988 |
| `oracle_monoculture|fraction=0.34|multiplier=1` | 100.00% | +0.00% | 35.04% | 72.22% | 0.092 | 40913 |
| `oracle_monoculture|fraction=0.67|multiplier=1` | 100.00% | +0.00% | 35.04% | 72.22% | 0.092 | 40913 |
| `oracle_monoculture|fraction=1|multiplier=1` | 100.00% | +0.00% | 35.04% | 72.22% | 0.092 | 40913 |
| `oracle_invasion|fraction=0|multiplier=1` | 99.04% | +2.83% | 25.98% | 0.84% | 0.072 | 30075 |
| `oracle_invasion|fraction=0.34|multiplier=1` | 97.64% | +7.05% | 37.30% | 64.23% | 0.100 | 25389 |
| `oracle_invasion|fraction=0.67|multiplier=1` | 97.64% | +7.05% | 37.30% | 64.23% | 0.100 | 25389 |
| `oracle_invasion|fraction=1|multiplier=1` | 97.64% | +7.05% | 37.30% | 64.23% | 0.100 | 25389 |

## Falsification verdict

- Cleared: **True**
- Cells clearing the test: `7`

The oracle arms are harness-local diagnostic upper bounds, not shipped reporter policies. A cell clears the test when the correct-report rate rises across the run above the static-prior null without changing initial parameters, or when parent–child reproductive correlation exceeds 0.2.
