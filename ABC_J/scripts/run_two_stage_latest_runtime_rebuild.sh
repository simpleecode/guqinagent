#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${TEACHER_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python}"
INPUT="$ROOT/ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
OUTPUT="$ROOT/ABC_J/agent_training/two_stage_audited/teacher_latest_runtime_raw"
cd "$ROOT"
"$PYTHON" - "$INPUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import os
from ABC_J.scripts.generate_teacher_tool_trajectories import validate_normalized_tuning
load_dotenv('.env')
assert os.getenv('GLM_API_KEY') or os.getenv('ANTHROPIC_API_KEY'), 'teacher credentials missing'
p = Path(sys.argv[1])
report_path=Path('ABC_J/agent_training/two_stage_audited/recapture_phrase_replacement_report.json')
report=json.loads(report_path.read_text()) if report_path.exists() else {}
expected_hash=report.get('selected_input_sha256','951bb201abfaa684bd55da9d87eb9f6b95796ffc3bf81bd42b5df6f2f2b22404')
expected_count=report.get('new_selected_total',3995)
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected_hash, 'input changed; recheck before resume'
ids=set(); scores=Counter()
test={json.loads(l)['score_key'] for l in open('train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl')}
for line in p.open():
    r=json.loads(line); validate_normalized_tuning(r)
    assert r['trajectory_id'] not in ids
    ids.add(r['trajectory_id']); scores[r['score_key']]+=1
assert len(ids)==expected_count and not set(scores)&test
assert not {'S0ejf3Pz','SBFpPuMq','S9j70QqA','SqMgYYGH'}&set(scores)
print(json.dumps({'preflight':'passed','phrases':len(ids),'scores':len(scores),'test_overlap':0}),flush=True)
PY
TARGET_COUNT="$("$PYTHON" -c 'import json; from pathlib import Path; p=Path("ABC_J/agent_training/two_stage_audited/recapture_phrase_replacement_report.json"); print(json.loads(p.read_text())["new_selected_total"] if p.exists() else 3995)')"
exec "$PYTHON" ABC_J/scripts/run_teacher_batch_parallel.py \
  --input "$INPUT" --output-dir "$OUTPUT" --target-count "$TARGET_COUNT" \
  --workers 7 --balance-by-score --model glm-5.3 \
  --min-interval 0.5 --max-tool-rounds 24 --max-attempts 3 \
  --allow-private-reasoning-leakage --include-guqinizer-no-op "$@"
