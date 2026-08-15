# Where the correctness payoff path is severed

Reproducible with `scripts/measure_payoff_coupling.py`; generated tables in
`docs/payoff-coupling.md` / `docs/payoff-coupling.json`. Instrument:
`SparseSensorScenario` (24 candidate locations, 83 distinct event locations,
3.00% static-prior null, 4.17% uniform null), 200 steps, seeds 42–46,
`grounded_input_fraction=0.67` held fixed, initial population 20, cap 60.

Grounded-stream access (#55) removed input starvation, but neither falsification
clause cleared: best within-run drift was +0.87 pp and parent–child reproductive
correlation stayed at 0.05–0.13. This measurement asks the prior question — does
being correct change an agent's currency balances and its offspring count at all —
by recording, per agent, correctness alongside every currency inflow and its
realized offspring, and correlating each link of the chain separately.

## Per-link coupling (Pearson r, mean over 5 seeds)

The `oracle_invasion` arm seeds 15% of the population with a
correct-by-construction reporter, which is what gives the correctness axis enough
spread to measure; `ordinary` is the evolved population.

| Link | `ordinary` | `oracle_invasion` | Verdict |
|---|---:|---:|---|
| correctness → user trust | −0.068 | **+0.821** | intact |
| precision → user trust | −0.073 | **+0.716** | intact |
| trust → attention income | +0.135 | +0.058 | **severed (scale)** |
| correctness → attention income | +0.032 | +0.030 | **severed** |
| correctness → information income | −0.012 | +0.102 | absent by design |
| attention income → offspring | +0.348 | +0.483 | intact |
| information income → offspring | +0.085 | +0.071 | weak |
| correctness → offspring | −0.013 | +0.053 | **severed** |
| report volume → offspring | −0.287 | +0.045 | reporting is net-costly |

Verification works: a 97.6%-correct oracle earns the trust it should
(r = +0.82). Reproduction responds to attention (r = +0.48). The break is in
between — trust does not convert into attention income at a magnitude that
matters, so a near-perfect reporter ends the run with no offspring advantage
(1.32 vs 0.92 for never-correct agents, inside seed noise).

## Why the middle link fails: currency scale

| Quantity | `ordinary` | `oracle_invasion` |
|---|---:|---:|
| Attention income / agent-step | 0.0250 | 0.0208 |
| Information income / agent-step | 0.7836 | 0.8013 |
| Information share of final reserves | 116.7% | 105.0% |
| Peer-subsidy share of information income | 18.8% | 14.5% |
| One false alarm, in agent-steps of attention income | 12.3 | 14.4 |
| Attention income, silent adults | 0.0248 | 0.0218 |
| Attention income, reporting adults | 0.0254 | 0.0205 |
| Mean offspring, silent adults | 1.47 | 1.38 |
| Mean offspring, reporting adults | 1.00 | 1.25 |

1. **The correctness-sensitive currency is ~30× smaller than the
   correctness-blind one.** `allocate_attention` divides a fixed total user
   budget (1.0 on this instrument) across the whole living population, so
   attention income is ~0.02/agent-step regardless of merit, while information
   income — compression yield plus a peer-residual subsidy, neither of which
   depends on correctness — runs ~0.78/agent-step and supplies >100% of the
   reserve that `Agent.can_reproduce` tests via `energy.total`.
2. **Silence pays the same as reporting.** Attention is allocated on
   `trust × relevance` of an agent's signal vector, not on having filed
   anything, so non-reporters draw the same ~0.02/step. Reporting only adds
   downside: `false_alarm_penalty` (0.3) costs 12–14 agent-steps of total
   attention income per false alarm, while a correct report buys +0.05 trust,
   worth ~0.01/step of income.
3. **The trust asymmetry prices honesty out of reach of this instrument.**
   `trust_delta_pos=0.05` vs `trust_delta_neg=0.2` puts the break-even precision
   for reporting at **80%**, while the instrument's own decoder ceiling is 16%
   and evidence inferability is 24%. No honest evolved reporter on this
   instrument can be right often enough for reporting to be trust-positive, so
   the gradient points at silence — and indeed silent adults out-reproduce
   reporting ones in the evolved arm (1.47 vs 1.00).

## Causal check, and a second gate behind the first

Scaling every user's attention budget 20× (diagnostic override in the script;
no engine change) restores the severed middle link, confirming it is a scale
problem rather than a wiring problem — but exposes a second severed link
immediately behind it. `oracle_invasion`, 200 steps, seeds 42–44:

| Budget scale | False-alarm penalty | correctness → attn income | correctness → offspring | attention-limited agent-steps | steps at population cap |
|---:|---:|---:|---:|---:|---:|
| 1× | 0.3 | +0.015 | +0.052 | 88.3% | 23.0% |
| 1× | 0.0 | +0.102 | +0.003 | 91.2% | 23.5% |
| 20× | 0.3 | **+0.425** | +0.143 | 17.2% | 95.5% |
| 20× | 0.0 | **+0.594** | +0.063 | 16.5% | 96.7% |

Once attention is abundant enough for correctness to register in income, nearly
every agent becomes solvent (co-limited agent-steps fall 98% → 21–24%) and the
**population cap becomes the rationer**: 95–97% of steps sit at the cap, and
`attempt_reproduction` iterates `eligible` in agent-creation order and `break`s
at `max_population`, so surplus merit is discarded by arrival order rather than
converted into offspring. Correctness → offspring stays ≤ +0.143.

## Conclusion

Two links are severed, in series:

1. **trust → attention income**: the only correctness-sensitive currency is a
   fixed zero-sum budget an order of magnitude below reproduction costs, is paid
   for signal relevance rather than for reporting, and is charged an asymmetric
   false-alarm penalty whose break-even precision (80%) exceeds the instrument's
   decoder ceiling (16%).
2. **income → differential offspring under a binding cap**: reproduction
   opportunities are rationed by creation order at `max_population`, so extra
   income does not become extra offspring once the cap binds.

Fixing (1) without (2) moves the bottleneck rather than closing it. Neither
falsification clause is cleared by this measurement, and nothing here changes
engine mechanics: the ledger and the two overrides are diagnostics only.
