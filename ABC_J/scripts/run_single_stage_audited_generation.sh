#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/F/work/Gunqin-agent

ROOT="ABC_J/agent_training"
INPUT="$ROOT/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
OUTPUT="$ROOT/single_stage_audited/teacher_raw"

mkdir -p "$OUTPUT"

python ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT" \
  --target-count 5147 \
  --workers 7 \
  --score-shard-count 7 \
  --shard-by-trajectory-id \
  --worker-prefix single_stage \
  --model glm-5.3 \
  --min-interval 0.5 \
  --stage single_stage \
  --allow-private-reasoning-leakage \
  > "$OUTPUT/run.log" 2>&1
