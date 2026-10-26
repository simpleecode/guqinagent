#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
BUILD="$BASE/.walk_blank_rebuild"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python

cd "$ROOT"
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

base = Path('ABC_J/agent_training/two_stage_audited')
build = base / '.walk_blank_rebuild'
report = json.loads((build / 'selection_report.json').read_text(encoding='utf-8'))
source = base / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
digest = hashlib.sha256()
with source.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == report['source_sha256'], 'Source changed'
ids = (build / 'trajectory_ids.txt').read_text(encoding='utf-8').splitlines()
assert len(ids) == len(set(ids)) == report['selected_phrases'] == 62
assert set(ids) == set(report['selected_ids'])
worker_root = build / 'regenerated_raw/.parallel_workers'
if worker_root.exists():
    assigned = []
    for index in range(7):
        id_file = worker_root / f'walk_{index:02d}/trajectory_ids.txt'
        if id_file.exists():
            assigned.extend(id_file.read_text(encoding='utf-8').splitlines())
    assert len(assigned) == len(ids) and set(assigned) == set(ids)
print(f'Validated {len(ids)} selected phrases and source SHA.', flush=True)
PY

SHARD_ARGS=(--balance-by-score)
if [[ -d "$BUILD/regenerated_raw/.parallel_workers" ]]; then
  SHARD_ARGS+=(--resume-existing-shards)
fi

exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BUILD/regenerated_raw" \
  --trajectory-id-file "$BUILD/trajectory_ids.txt" \
  --target-count 3976 --workers 7 "${SHARD_ARGS[@]}" --worker-prefix walk \
  --model glm-5.3 --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
