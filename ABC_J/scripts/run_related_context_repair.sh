#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
cd "$ROOT"
"$PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('ABC_J/agent_training/two_stage_audited')
report=json.loads((base/'related_context_repair/affected_report.json').read_text())
assert report['affected_phrases']==177
for path,key in [(base/'pitch_eligible_train_test/inferred_trajectories_train.jsonl','source_sha256'),
                 (base/'teacher_train_final/messages_train.jsonl','source_messages_sha256')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report[key],f'Input changed: {path}'
print('Preflight passed: 177 affected phrases; original final dataset and source unchanged.',flush=True)
PY
test ! -e "$BASE/related_context_repair/regenerated_raw"
exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BASE/related_context_repair/regenerated_raw" \
  --trajectory-id-file "$BASE/related_context_repair/trajectory_ids.txt" \
  --target-count 3976 --workers 7 --balance-by-score --model glm-5.3 \
  --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
