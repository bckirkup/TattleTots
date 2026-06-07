# TattleTots — Greenfield Build Plan

## Current State
Repo is empty: LICENSE (Apache 2.0), CODE_OF_CONDUCT.md, CONTRIBUTING.md only.

---

## Architecture Decision: Python 3.11+, Pydantic v2, NumPy/SciPy core

Consistent with your other repos (Crusher_to_the_Bridge, shipbiome-core). Engine is domain-agnostic; domain adapters live in separate repos.

---

## Phase 1: Scaffolding & CI (this PR)

| Task | Deliverable |
|------|-------------|
| Project structure | `src/tattletots/` package with `__init__.py`, `py.typed` |
| Build config | `pyproject.toml` (hatch/pip-installable), ruff lint, mypy strict |
| CI pipeline | `.github/workflows/ci.yml` — lint, typecheck, pytest on push/PR |
| Pre-commit | `.pre-commit-config.yaml` (ruff, mypy) |
| README | Architecture overview, quickstart, API sketch |
| SKILL file | `.agents/skills/tattletots/SKILL.md` |

## Phase 2: Core Domain Models

| Module | Responsibility |
|--------|---------------|
| `models/genome.py` | Agent genome: model class, hyperparams, input prefs, escalation threshold, target user |
| `models/agent.py` | Agent state: dual energy reserves, input connections, compression model, lifecycle stage |
| `models/stream.py` | Data stream abstraction: raw & residual, multivariate time series |
| `models/user.py` | Human user: attention budget, priority vector, trust state per agent |
| `models/energy.py` | Energy accounting: information yield, attention income, costs, penalties |
| `models/report.py` | Escalation report: signal, confidence, target user, timestamp |

## Phase 3: Simulation Engine Core

| Module | Responsibility |
|--------|---------------|
| `engine/world.py` | World state: population, streams, users, trophic graph, clock |
| `engine/step.py` | Single time-step orchestration (order of operations) |
| `engine/compression.py` | Agent compression models (PCA, AR, wavelet, etc.) — pluggable |
| `engine/trophic.py` | Trophic attachment: agents choose inputs maximizing metabolic yield |
| `engine/energy_accounting.py` | Dual-currency bookkeeping per step |
| `engine/death.py` | Starvation check, agent removal, recycling |

## Phase 4: Attention & Trust

| Module | Responsibility |
|--------|---------------|
| `engine/attention.py` | Softmax attention allocation across agents per user |
| `engine/trust.py` | Trust update on verified outcomes (asymmetric: hard to build, easy to destroy) |
| `engine/escalation.py` | Agent decision to escalate, false-alarm penalty mechanics |

## Phase 5: Reproduction & Evolution

| Module | Responsibility |
|--------|---------------|
| `engine/reproduction.py` | Energy threshold → spawn offspring |
| `engine/mutation.py` | Genome mutations: model type, hyperparams, input prefs, thresholds |
| `engine/recombination.py` | Sexual recombination (two-parent crossover) |
| `engine/development.py` | Juvenile → adult lifecycle with development duration |

## Phase 6: Advanced Mechanics

| Module | Responsibility |
|--------|---------------|
| `engine/whistleblowing.py` | Agent consuming another's output to detect dishonesty |
| `engine/domestication.py` | Downstream→upstream signal for shaping compression behavior |
| `engine/competition.py` | Niche overlap, competitive exclusion, cosine-similarity attention competition |

## Phase 7: Smoke Test & Validation

| Deliverable | Description |
|-------------|-------------|
| `scenarios/gaussian_shift.py` | Built-in: K=10 structured Gaussian, noise, midpoint shift, 2 users |
| `tests/test_smoke.py` | Automated validation of 5 success criteria from requirements §9 |
| `tests/test_math_properties.py` | Tests for §6 mathematical invariants (entropy decrease, chain depth bounds, etc.) |

## Phase 8: Telemetry & Domain Interface

| Module | Responsibility |
|--------|---------------|
| `interface/domain_adapter.py` | ABC for domain repos (generate streams, provide ground truth, score relevance) |
| `telemetry/recorder.py` | Trophic topology history, energy flows, demographics, detection performance |
| `telemetry/cost_accounting.py` | Surveillance + response + damage cost tracking |

## Phase 9: Documentation & Polish

| Deliverable | Description |
|-------------|-------------|
| `docs/architecture.md` | Full system architecture with diagrams |
| `docs/theory.md` | Dual-currency model (from your attached doc, canonical reference) |
| `docs/domain_integration.md` | How to write a domain adapter |
| `CHANGELOG.md` | Release notes |

## Phase 10 (future): Dashboard

Small graphical dashboard (Streamlit or similar) — not priority for this session.

---

## Mathematical Invariants to Test

1. Residual entropy decreases through chains
2. Chain depth bounded by signal rank (~ceil(K/k))
3. Branching topologies more stable than linear
4. H-D-W equilibrium (honest-deceiver-whistleblower coexistence)
5. Domestication improves yield only when signals overlap
6. Attention zero-sum per user (hard constraint)

## Success Criteria (Engine Passes If)

1. Trophic hierarchies depth > 2 emerge from random seed
2. Population reaches stable equilibrium (births ≈ deaths)
3. Removing basal streams → upstream extinction cascades
4. False-alarm agents die; accurate agents reproduce
5. At least 2 distinct genome clusters coexist at equilibrium

---

## What I'll Build in This Session

**Phases 1–7** — full core engine with passing CI, comprehensive tests, and the smoke test scenario. This gives you a working simulation you can run from CLI and connect to domain repos.

## What Needs Your Input

1. **Python version preference?** I'll default to 3.11+ (matches Crusher_to_the_Bridge).
2. **Domain adapter priority?** Which domain repo to wire up first (FireEcology? CruiseEcology? Or just the built-in Gaussian for now)?
3. **Any model types you want in the initial gene pool?** I'll start with PCA, AR(1), and a simple threshold detector — extensible.
