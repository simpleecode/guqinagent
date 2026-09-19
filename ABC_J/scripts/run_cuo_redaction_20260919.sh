#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${TEACHER_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python}"
BASE="$ROOT/ABC_J/agent_training/cuo_double_stopped_rerun_20260919"
INPUT="$BASE/raw_merged"
WORKERS="$BASE/redaction_workers"
mkdir -p "$WORKERS"

pids=()
for shard in 0 1 2 3 4 5 6; do
  out="$WORKERS/worker_${shard}"
  log="$WORKERS/worker_${shard}.log"
  nohup "$PYTHON" "$ROOT/scripts/redact_teacher_reasoning.py" \
    --input-dir "$INPUT" \
    --output-dir "$out" \
    --model glm-5.3 \
    --shard-count 7 \
    --shard-index "$shard" >"$log" 2>&1 < /dev/null &
  pids+=("$!")
  echo "started redaction shard $shard (pid $!), log: $log"
done
wait "${pids[@]}"
