#!/usr/bin/env bash
# Ablate the two payoff-path fixes independently on SparseSensor.
# Run from the repository root. Writes docs/payoff-coupling.{json,md} on the last cell only.
set -euo pipefail

STEPS=${STEPS:-200}
SEEDS=${SEEDS:-"42 43 44 45 46"}
OUT=${OUT:-payoff-fix-ablation}
mkdir -p "${OUT}"

run() {
  local name=$1
  shift
  uv run --no-sync --no-build python scripts/measure_payoff_coupling.py \
    --steps "${STEPS}" --seeds ${SEEDS} --no-write "$@" >"${OUT}/${name}.md"
}

run baseline
run value_only --config correct_report_attention_value=8.0
run merit_only --config reproduction_merit_ordering=true
run value_and_merit --config correct_report_attention_value=8.0 reproduction_merit_ordering=true
run value_and_merit_symmetric_trust --config correct_report_attention_value=8.0 \
  reproduction_merit_ordering=true trust_delta_neg=0.05 false_alarm_penalty=0.05
