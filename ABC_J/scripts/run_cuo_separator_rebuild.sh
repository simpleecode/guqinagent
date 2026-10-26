#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/ABC_J/agent_training/two_stage_audited"
BUILD="$BASE/.cuo_separator_rebuild"
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python

cd "$ROOT"
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

base = Path('ABC_J/agent_training/two_stage_audited')
build = base / '.cuo_separator_rebuild'
report = json.loads((build / 'selection_report.json').read_text(encoding='utf-8'))
for relative, expected in (
    ('pitch_eligible_train_test/inferred_trajectories_train.jsonl', report['source_sha256']),
    ('teacher_train_final/teacher_trajectory_audit.jsonl', report['audit_sha256']),
):
    h = hashlib.sha256()
    with (base / relative).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == expected, f'Source changed: {relative}'
ids = (build / 'trajectory_ids.txt').read_text(encoding='utf-8').splitlines()
assert len(ids) == len(set(ids)) == report['selected_phrases'] == 226
worker_root = build / 'regenerated_raw/.parallel_workers'
if worker_root.exists():
    assigned = []
    for index in range(7):
        id_file = worker_root / f'cuo_{index:02d}/trajectory_ids.txt'
        assert id_file.exists(), f'Missing existing shard: {id_file}'
        assigned.extend(id_file.read_text(encoding='utf-8').splitlines())
    assert set(assigned) == set(ids) and len(assigned) == len(ids), 'Existing shards differ'
print(f'Validated {len(ids)} targeted phrases and immutable source hashes.', flush=True)
PY

SHARD_ARGS=(--balance-by-score)
if [[ -d "$BUILD/regenerated_raw/.parallel_workers" ]]; then
  SHARD_ARGS+=(--resume-existing-shards)
fi

exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$BASE/pitch_eligible_train_test/inferred_trajectories_train.jsonl" \
  --output-dir "$BUILD/regenerated_raw" \
  --trajectory-id-file "$BUILD/trajectory_ids.txt" \
  --target-count 3976 --workers 7 "${SHARD_ARGS[@]}" --worker-prefix cuo \
  --model glm-5.3 --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op
