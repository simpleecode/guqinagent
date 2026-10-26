#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
BUILD="$BASE/.final_warning_rebuild"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python
cd "$ROOT"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from ABC_J.scripts.prepare_final_warning_rebuild import sha256, BASE, ROOT, OUT
r=json.loads((OUT/'selection_report.json').read_text())
assert sha256(BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl')==r['source_sha256']
assert sha256(BASE/'teacher_train_final/teacher_trajectory_audit.jsonl')==r['audit_sha256']
assert all(sha256(ROOT/f)==v for f,v in r['runtime_sha256'].items())
ids=(OUT/'trajectory_ids.txt').read_text().splitlines()
assert len(ids)==len(set(ids))==r['selected_phrases'] and set(ids)==set(r['selected_ids'])
assert ids, 'No warning phrases to rebuild'
print(f'Validated {len(ids)} warning phrases and source/runtime hashes.',flush=True)
PY
SHARD_ARGS=(--balance-by-score)
if [[ -d "$BUILD/regenerated_raw/.parallel_workers" ]]; then
  SHARD_ARGS+=(--resume-existing-shards)
fi
exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BUILD/regenerated_raw" --trajectory-id-file "$BUILD/trajectory_ids.txt" \
  --target-count 3976 --workers 7 "${SHARD_ARGS[@]}" --worker-prefix finalwarning \
  --model glm-5.3 --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
