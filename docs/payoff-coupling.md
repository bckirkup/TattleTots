# Does correctness pay? Per-link payoff coupling

- Adapter: `tattletots.scenarios.sparse_sensor:SparseSensorScenario`
- Steps per run: `200`
- Seeds: `42, 43, 44, 45, 46`
- Grounded input fraction (fixed): `0.67`
- User attention-budget scale: `1`
- False-alarm penalty override: `None`

## Coupling of each link (Pearson r, mean over seeds)

| Link | `ordinary` | `oracle_invasion` |
|---|---|---|
| correctness -> user trust | -0.068 | +0.821 |
| precision -> user trust | -0.073 | +0.716 |
| trust -> attention income | +0.135 | +0.058 |
| correctness -> attention income | +0.032 | +0.030 |
| correctness -> information income | -0.012 | +0.102 |
| attention income -> offspring | +0.348 | +0.483 |
| information income -> offspring | +0.085 | +0.071 |
| correctness -> offspring | -0.013 | +0.053 |
| report volume -> attention income | +0.018 | +0.031 |
| report volume -> offspring | -0.287 | +0.045 |

## Currency scale and rationing

| Quantity | `ordinary` | `oracle_invasion` |
|---|---|---|
| Correct-report rate | 12.80% | 97.64% |
| Attention income / agent-step | 0.0250 | 0.0208 |
| Information income / agent-step | 0.7836 | 0.8013 |
| Information share of reserves | 116.66% | 104.97% |
| Peer-subsidy share of info income | 18.79% | 14.46% |
| Mean offspring, ever-correct agents | 1.30 | 1.32 |
| Mean offspring, never-correct agents | 0.95 | 0.92 |
| Silent adults (never reported) | 450.2 | 144.6 |
| Attention income, silent adults | 0.0248 | 0.0218 |
| Attention income, reporting adults | 0.0254 | 0.0205 |
| Mean offspring, silent adults | 1.47 | 1.38 |
| Mean offspring, reporting adults | 1.00 | 1.25 |
| Trust break-even precision | 80.00% | 80.00% |
| False alarm cost (agent-steps of attention income) | 12.3 | 14.4 |

### Reproduction gating (share of agent-steps)

| Condition | `ordinary` | `oracle_invasion` |
|---|---|---|
| eligible_share | 44.54% | 59.58% |
| co_limited_share | 96.57% | 98.61% |
| attention_limited_share | 81.23% | 88.20% |
| information_limited_share | 15.34% | 10.40% |
| population_capped_step_share | 6.50% | 20.10% |
