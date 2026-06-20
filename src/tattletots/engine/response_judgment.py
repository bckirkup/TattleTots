"""Shared logic for post-dispatch responder necessity judgments."""

from __future__ import annotations

from tattletots.models.response_outcome import ResponseOutcome


def judge_necessity(
    before: float,
    after: float,
    *,
    problem_threshold: float = 0.0,
    min_reduction: float = 0.05,
) -> tuple[bool, bool, bool]:
    """Return (problem_present, mitigated, response_necessary).

    A response is necessary when a problem existed before dispatch and severity
    dropped by at least ``min_reduction`` fraction afterward.
    """
    problem = before > problem_threshold
    mitigated = problem and after < before * (1.0 - min_reduction)
    return problem, mitigated, problem and mitigated


def aggregate_outcomes(outcomes: list[ResponseOutcome]) -> dict[str, int]:
    """Summarize response outcomes for telemetry."""
    dispatched = sum(1 for o in outcomes if o.dispatched)
    necessary = sum(1 for o in outcomes if o.response_necessary)
    unnecessary = sum(1 for o in outcomes if o.dispatched and not o.response_necessary)
    return {
        "responses_dispatched": dispatched,
        "responses_judged_necessary": necessary,
        "responses_judged_unnecessary": unnecessary,
    }


def patch_step_record_responses(
    history: list[object],
    outcomes: list[ResponseOutcome],
) -> None:
    """Update the most recent StepRecord with response outcome counts."""
    if not history:
        return
    counts = aggregate_outcomes(outcomes)
    last = history[-1]
    for key, value in counts.items():
        setattr(last, key, value)
