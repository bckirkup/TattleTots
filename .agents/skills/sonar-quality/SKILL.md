---
name: sonar-quality
description: Prevent SonarCloud/SonarQube issues when writing or changing code in the TattleTots repo, and resolve findings when they are reported.
---

# SonarCloud Quality Standards

This repository is analyzed by SonarCloud on every push. Follow these rules so new code
does not reintroduce flagged patterns.

## Pre-commit checklist

```bash
pre-commit run --all-files
python scripts/sonar_guard.py src tests scripts "Large Experiments"
ruff check src/ tests/ scripts/ "Large Experiments/"
ruff format --check src/ tests/ scripts/ "Large Experiments/"
pytest
```

After substantive changes, verify the SonarCloud check on the PR or compare against
`https://sonarcloud.io/project/issues?id=bckirkup_TattleTots&issueStatuses=OPEN,CONFIRMED`.

## Rule catalog (current TattleTots findings and local defenses)

The Ruff configuration enables `ARG`, `C90` (maximum complexity 15), `NPY`, and
the mechanical Bandit rules that are useful here. `scripts/sonar_guard.py`
provides the local checks for float equality in asserts, bare NumPy randomness,
and uncommented `pass` stubs. The `zizmor` pre-commit hook and the `workflows`
CI job check GitHub Actions files.

### python:S1244 — no floating-point equality

Never use `==` or `!=` on `float` values. Use `pytest.approx` in tests or
`math.isclose` in production code.

```python
# Bad
assert result.cost == 0.0

# Good (tests)
assert result.cost == pytest.approx(0.0)

# Good (src)
assert math.isclose(result.cost, 0.0, rel_tol=1e-9, abs_tol=1e-12)
```

### python:S1172 — unused function parameters

Prefix intentionally unused parameters with `_`. This applies to protocol stubs,
callback hooks, and interface methods where a parameter is required by signature
but not used in this implementation.

```python
def setup(self, _adapter: Any, _run: RunContext) -> dict[str, Any]:
    return {}
```

### python:S1186 — empty function bodies

Empty stub methods must include an inline comment explaining why they are empty.

```python
def write_output(self, result, path) -> None:
    pass  # Stub hook: output persistence not exercised in this test.
```

### python:S6709 — reproducible randomness

Construct RNGs with an explicit seed. Store the generator on the object; do not call
bare `np.random.*` without a seeded `Generator`.

```python
self._rng = np.random.default_rng(seed)
value = self._rng.uniform(0, 1)
```

### python:S6711 — prefer numpy.random.Generator

Use `np.random.default_rng(seed)` instead of legacy `np.random.seed()` /
`np.random.RandomState` / module-level `np.random.*`.

### python:S3776 — cognitive complexity ≤ 15

When Sonar flags a function, extract helpers for distinct logical blocks. Prefer
early returns over deep nesting. Each helper should have a single responsibility.

### pythonsecurity:S6680, S6639, S6549 — taint-analysis security findings

These are inter-procedural taint rules. The local Ruff and guard checks can
enforce safer conventions, but they cannot prove that untrusted data reaches
or avoids a sink. Only the CI Sonar job can catch these findings reliably.

### githubactions:S8541, S8544 — workflow action pinning and hygiene

Actions in `.github/workflows` must use full commit SHAs with a version comment.
Run `zizmor` locally through pre-commit; CI repeats the check in the `workflows`
job. The `pip install --only-binary :all:` safeguard for published packages is
not caught by zizmor 1.29.0; it remains a CI-Sonar-only check.

### pythonsecurity:S8707 — CLI path traversal

Never pass raw user CLI paths to `open()`, `Path()`, or `os.path.join()` without
validation. Resolve paths and ensure they stay within an allowed base directory.

```python
def _safe_output_path(raw: str, base: Path) -> Path:
    resolved = (base / raw).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes output directory: {raw}")
    return resolved
```

### text:S8565 — predictable dependency versions

Keep `uv.lock` committed alongside `pyproject.toml`. Regenerate after dependency
changes:

```bash
uv lock
```

## Anti-patterns

| Pattern | Sonar rule | Fix |
|---|---|---|
| `assert x == 0.5` on floats | S1244 | `pytest.approx` / `math.isclose` |
| Unused hook parameter `adapter` | S1172 | Rename to `_adapter` |
| Bare `np.random.uniform()` | S6709, S6711 | `self._rng = np.random.default_rng(seed)` |
| `pass` with no comment in stub | S1186 | Add `# reason` on same line |
| `open(args.output)` from argparse | S8707 | Validate with `_safe_output_path` |
| Unpinned deps, no lock file | S8565 | Commit `uv.lock` |

## When Sonar reports issues on existing code

1. Fetch the issue list from SonarCloud or the PR check summary.
2. Fix mechanical rules first (S1244, S1172, S6709, S1186, S1481).
3. Refactor S3776 hotspots by extracting named helpers.
4. Add path guards for S8707 in CLI/baseline scripts.
5. Re-run local lint + tests before pushing.

## Validating the local guard layers (adversarial self-test)

The guards are only useful if they actually reject bad code. To verify (or after
changing `scripts/sonar_guard.py`, `.pre-commit-config.yaml`, or `.github/workflows/ci.yml`),
plant one defect of each class in scratch files, confirm the rejection, then delete them
and confirm the tree is clean — never commit a planted defect.

```bash
# guard classes (expect exit 1 and a "path:line:1: RULE message" line)
python scripts/sonar_guard.py <scratch file>       # S1244 / S6709 / S1186
ruff check <scratch file>                          # ARG001, C901 "is too complex (N > 15)", NPY002
zizmor --no-progress <scratch workflow>            # error[unpinned-uses] for an @vN tag, exit 14
git add -A && git commit -m scratch                # must be REFUSED by pre-commit
```

`pre-commit run --all-files` takes ~1-4 min on a warm cache; the mypy hook builds an
isolated Python 3.11 env, so allow extra time on the first run.

### Known blind spots of scripts/sonar_guard.py (as of PR #42)

The guard is deliberately conservative and AST-literal based. It does NOT flag:

- `assert a == b` where both sides are float-typed *variables* (no float literal operand).

Treat these as gaps to close if Sonar reports findings the local guard missed, not as
reasons to distrust the guard.

The guard's `_comment_lines` helper preserves comments collected before a
`tokenize.TokenError` and returns them after the partial token stream is consumed.
Both the pre-commit hook and CI guard scan `src`, `tests`, `scripts`, and
`Large Experiments`.

### Environment-dependent failures (expected, do not "fix")

- `tests/test_determinism.py::test_seeded_run_has_golden_fingerprint` fails on machines whose
  NumPy/BLAS differs from the one that produced the golden fingerprint. Confirm by running it
  in a clean worktree of the merge base before investigating.
- `mypy src/` run directly from a Python 3.12 virtualenv fails inside
  `numpy/__init__.pyi` ("Type statement is only supported in Python 3.12 and greater").
  Use the pre-commit mypy hook (isolated 3.11 env) as the authoritative typecheck locally.
