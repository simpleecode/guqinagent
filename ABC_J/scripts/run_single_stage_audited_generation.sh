#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="ABC_J/agent_training"
INPUT="$DATA_ROOT/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
OUTPUT="$DATA_ROOT/.single_stage_rebuild/teacher_raw"
PYTHON="${TEACHER_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python}"
TARGET_COUNT="$(wc -l < "$INPUT" | tr -d ' ')"

mkdir -p "$OUTPUT"
RESUME_ARGS=()
if [[ -f "$OUTPUT/.parallel_workers/current_00/trajectory_ids.txt" ]]; then
  RESUME_ARGS+=(--resume-existing-shards)
fi

"$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT" \
  --target-count "$TARGET_COUNT" \
  --workers 7 \
  --score-shard-count 7 \
  --balance-by-score \
  --worker-prefix current \
  --model glm-5.3 \
  --min-interval 0.5 \
  --stage single_stage \
  --allow-private-reasoning-leakage \
  "${RESUME_ARGS[@]}" \
  > "$OUTPUT/run.log" 2>&1
