# Determinism float references

`determinism_float_references.json` records the float telemetry produced by the
default and legacy seeded runs in `tests/test_determinism.py`. The values are
reference data for tolerance-based comparisons, not golden correctness values.
The tests compare them with `pytest.approx(rel=1e-9)` because Python runtimes can
accumulate the same values one ULP apart.

To deliberately regenerate the file, run `_run_payload(42)` for both the
default configuration and the legacy configuration
(`reproduction_coupling_strength=0.0`, `grounding_quality_strength=0.0`), then
extract each record's float fields as the two top-level entries. Review the
resulting diff before replacing the file. Do not update the data merely to make
a failing test pass; update it only for an intentional simulation behavior
change.
