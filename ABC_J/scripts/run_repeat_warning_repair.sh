#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=ABC_J/agent_training/two_stage_audited
exec /opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BASE/repeat_warning_dependency_repair/regenerated_raw" \
  --trajectory-id-file "$BASE/repeat_warning_dependency_repair/trajectory_ids.txt" \
  --target-count 3976 --workers 7 --balance-by-score --model glm-5.3 \
  --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
