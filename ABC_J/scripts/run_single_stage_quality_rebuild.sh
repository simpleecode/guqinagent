#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/ABC_J/agent_training/single_stage_audited/.quality_rebuild"
PYTHON="${TEACHER_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python}"
"$PYTHON" - <<'PY'
import json,shutil,hashlib
from pathlib import Path
root=Path.cwd();out=root/'ABC_J/agent_training/single_stage_audited/.quality_rebuild'
r=json.loads((out/'selection_report.json').read_text());snapshot=out/'runtime_snapshot'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl')==r['source_sha256']
assert sha(root/'ABC_J/agent_training/single_stage_audited/teacher_raw/teacher_trajectory_audit.jsonl')==r['audit_sha256']
ids=(out/'trajectory_ids.txt').read_text().splitlines()
assert ids and len(ids)==len(set(ids))==r['selected_phrases'] and set(ids)==set(r['selected_ids'])
if not snapshot.exists():
    assert all(sha(root/f)==v for f,v in r['runtime_sha256'].items())
    snapshot.mkdir()
    for directory in ('agents','scripts','skills','ABC_J/scripts'):
        shutil.copytree(root/directory,snapshot/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Read credentials at execution time without copying them into artifacts.
    (snapshot/'.env').symlink_to(root/'.env')
assert all(sha(snapshot/f)==v for f,v in r['runtime_sha256'].items())
print('Verified selected IDs and immutable generation runtime snapshot',len(ids),flush=True)
PY
RESUME=(--balance-by-score)
if [[ -d "$BUILD/regenerated_raw/.parallel_workers" ]]; then
  RESUME+=(--resume-existing-shards)
fi
exec "$PYTHON" "$BUILD/runtime_snapshot/ABC_J/scripts/run_teacher_batch_parallel.py" \
  --input "$ROOT/ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BUILD/regenerated_raw" --trajectory-id-file "$BUILD/trajectory_ids.txt" \
  --target-count 3976 --workers 7 --worker-prefix quality \
  --stage single_stage --model glm-5.3 --min-interval 0.5 \
  --max-tool-rounds 24 --max-attempts 3 --allow-private-reasoning-leakage "${RESUME[@]}"
