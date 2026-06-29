---
name: sonar-quality
description: Resolve and prevent SonarCloud/SonarQube code quality issues in Python repos. Use when fixing Sonar findings, reviewing PRs with Sonar annotations, or writing new code that must pass SonarCloud analysis.
---

# SonarCloud Quality Standards

This repository is analyzed by SonarCloud on every push. Follow these rules so new code
does not reintroduce flagged patterns.

## Pre-commit checklist

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest
```

After substantive changes, verify the SonarCloud check on the PR or compare against
`https://sonarcloud.io/project/issues?id=bckirkup_TattleTots&issueStatuses=OPEN,CONFIRMED`.

## Rule catalog (most common in this monorepo)

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
