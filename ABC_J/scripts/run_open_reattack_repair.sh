#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python
BASE=ABC_J/agent_training/two_stage_audited
test -s "$BASE/ordinary_open_reattack_repair/trajectory_ids.txt"
RESUME_ARGS=()
if test -f "$BASE/ordinary_open_reattack_repair/regenerated_raw/.parallel_workers/worker_00/trajectory_ids.txt"; then
  RESUME_ARGS+=(--resume-existing-shards)
fi
exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BASE/ordinary_open_reattack_repair/regenerated_raw" \
  --trajectory-id-file "$BASE/ordinary_open_reattack_repair/trajectory_ids.txt" \
  --target-count 3976 --workers 7 --balance-by-score --model glm-5.3 \
  --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op "${RESUME_ARGS[@]}"
