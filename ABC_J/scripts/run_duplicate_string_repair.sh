#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
REPAIR_NAME=${REPAIR_NAME:-duplicate_string_repair}
export REPAIR_NAME
cd "$ROOT"
"$PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
b=Path('ABC_J/agent_training/two_stage_audited')
name=os.environ['REPAIR_NAME']
assert name in {'duplicate_string_repair','duplicate_string_retry','duplicate_string_last_retry'}
r=json.loads((b/name/'affected_report.json').read_text())
for path,key in [(b/'pitch_eligible_train_test/inferred_trajectories_train.jsonl','source_sha256'),
                 (b/'teacher_train_final/messages_train.jsonl','messages_sha256')]:
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    assert h==r[key],f'Input changed: {path}'
print(f"Verified {r['affected_phrases']} affected phrases; current canonical dataset unchanged.",flush=True)
PY
test ! -e "$BASE/$REPAIR_NAME/regenerated_raw"
exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BASE/$REPAIR_NAME/regenerated_raw" \
  --trajectory-id-file "$BASE/$REPAIR_NAME/trajectory_ids.txt" \
  --target-count 3976 --workers "${REPAIR_WORKERS:-7}" --balance-by-score --model glm-5.3 \
  --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
