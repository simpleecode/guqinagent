#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
REPAIR="$BASE/daiqi_state_repair"
SOURCE="$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
cd "$ROOT"
check_source() {
  "$PYTHON" - "$SOURCE" "$REPAIR/affected_report.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
report=json.loads(Path(sys.argv[2]).read_text())
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()==report['source_sha256']
assert hashlib.sha256(Path('scripts/audit_jianpu_jianzi_pitch.py').read_bytes()).hexdigest()==report['auditor_sha256']
assert report['affected_phrases']==77
assert {'SzQNSg3J-p0007','SzQNSg3J-p0011'}<=report['cases'].keys()
print('Preflight passed: 77 affected phrases; source and auditor unchanged.',flush=True)
PY
}
check_source
"$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$SOURCE" --output-dir "$REPAIR/regenerated_raw" \
  --trajectory-id-file "$REPAIR/trajectory_ids.txt" --target-count 3976 \
  --workers 7 --balance-by-score --model glm-5.3 --min-interval 0.5 \
  --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
check_source
"$PYTHON" ABC_J/scripts/replace_daiqi_repaired_teachers.py \
  --canonical "$BASE/teacher_latest_runtime_raw" --repair "$REPAIR/regenerated_raw" \
  --ids "$REPAIR/trajectory_ids.txt" --report "$REPAIR/replacement_report.json"
"$PYTHON" ABC_J/scripts/statistics_teacher_final.py \
  --directory "$BASE/teacher_latest_runtime_raw" --source "$SOURCE"
"$PYTHON" ABC_J/scripts/visualize_final_warning_subset.py \
  --directory "$BASE/teacher_latest_runtime_raw" --source "$SOURCE" \
  --output-dir "$BASE/teacher_latest_runtime_raw/final_warning_visualization/visualization_sharded"
echo 'Repair and validated replacement finished. Old token statistics are a pre-repair snapshot; remeasure before training.'
