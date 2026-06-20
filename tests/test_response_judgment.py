"""Unit tests for post-dispatch responder necessity judgments."""

from __future__ import annotations

from tattletots.engine.response_judgment import aggregate_outcomes, judge_necessity
from tattletots.models.response_outcome import ResponseOutcome


class TestJudgeNecessity:
    def test_no_problem(self) -> None:
        problem, mitigated, necessary = judge_necessity(0.0, 0.0)
        assert not problem
        assert not mitigated
        assert not necessary

    def test_problem_without_mitigation(self) -> None:
        problem, mitigated, necessary = judge_necessity(10.0, 10.0)
        assert problem
        assert not mitigated
        assert not necessary

    def test_partial_mitigation(self) -> None:
        problem, mitigated, necessary = judge_necessity(10.0, 8.0)
        assert problem
        assert mitigated
        assert necessary

    def test_respects_problem_threshold(self) -> None:
        problem, mitigated, necessary = judge_necessity(8.0, 4.0, problem_threshold=10.0)
        assert not problem
        assert not mitigated
        assert not necessary

    def test_insufficient_reduction(self) -> None:
        problem, mitigated, necessary = judge_necessity(10.0, 9.8, min_reduction=0.05)
        assert problem
        assert not mitigated
        assert not necessary


class TestAggregateOutcomes:
    def _outcome(self, *, dispatched: bool, necessary: bool) -> ResponseOutcome:
        return ResponseOutcome(
            agent_id="a1",
            responder_user_id="u1",
            time_step=1,
            location=(0, 0),
            response_type="suppression",
            dispatched=dispatched,
            problem_severity_before=1.0,
            problem_severity_after=0.5 if necessary else 1.0,
            problem_present=True,
            mitigated=necessary,
            response_necessary=necessary,
        )

    def test_aggregate_counts(self) -> None:
        outcomes = [
            self._outcome(dispatched=True, necessary=True),
            self._outcome(dispatched=True, necessary=False),
            self._outcome(dispatched=False, necessary=False),
        ]
        counts = aggregate_outcomes(outcomes)
        assert counts == {
            "responses_dispatched": 2,
            "responses_judged_necessary": 1,
            "responses_judged_unnecessary": 1,
        }
