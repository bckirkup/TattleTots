# Determinism float references

`determinism_float_references.json` records the float telemetry produced by the
default and legacy seeded runs in `tests/test_determinism.py`. The values are
reference data for tolerance-based comparisons, not golden correctness values.
The tests compare them with `pytest.approx(rel=1e-9)` because Python runtimes can
accumulate the same values one ULP apart.

`determinism_structural_references.json` records the corresponding float-free
projection. It is a deliberately labeled change-detector: it catches discrete
changes in IDs, counts, flags, locations, and other non-float telemetry, but it
does not validate float telemetry.

`determinism_agent_traces.json` records the per-step, per-agent energy and
attention trace for the same seeded runs. It is tolerance-based diagnostic
reference data, not a correctness golden: the test uses `pytest.approx(rel=1e-9)`
and reports the first divergence plus its per-step growth curve.

All three reference files below were deliberately regenerated after
deterministic sign canonicalization was added for PCA SVD components and the
Gaussian-shift scenario's QR basis. Their current values therefore describe the
post-canonicalization behavior.

To deliberately regenerate the file, run `_run_payload(42)` for both the
default configuration and the legacy configuration
(`reproduction_coupling_strength=0.0`, `grounding_quality_strength=0.0`), then
extract each record's float fields for
`determinism_float_references.json` and apply `_structural_projection()` for
`determinism_structural_references.json`. For the per-agent diagnostic, run
`_run_payload(42, capture_agent_trace=True)` for both configurations and extract
the `agent_trace` entries into `determinism_agent_traces.json`. Review the
resulting diff before replacing any file. Do not update the data merely to make
a failing test pass; update it only for an intentional simulation behavior
change.
