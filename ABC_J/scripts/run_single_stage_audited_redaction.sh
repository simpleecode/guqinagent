#!/usr/bin/env bash
# Redact all completed single-stage teacher trajectories, then merge shards in
# source order. The raw source is immutable; the resulting directory is the
# only public training candidate produced by this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${TEACHER_PYTHON:-python}"
BASE="$ROOT/ABC_J/agent_training/.single_stage_rebuild"
INPUT="$BASE/teacher_raw"
WORKER_ROOT="$BASE/redaction_workers"
OUTPUT="$BASE/teacher_final_redacted"
WORKERS=7

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$WORKER_ROOT"

pids=()
for shard in $(seq 0 $((WORKERS - 1))); do
  worker="$WORKER_ROOT/worker_$shard"
  mkdir -p "$worker"
  "$PYTHON" "$ROOT/ABC_J/scripts/redact_teacher_reasoning.py" \
    --input-dir "$INPUT" \
    --output-dir "$worker" \
    --model glm-5.3 \
    --shard-count "$WORKERS" \
    --shard-index "$shard" \
    >"$WORKER_ROOT/worker_$shard.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "started redaction worker $shard: $pid"
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ROOT/scripts/merge_redaction_workers.py" \
  --source "$INPUT" \
  --worker-root "$WORKER_ROOT" \
  --output-dir "$OUTPUT"

cp "$INPUT/generation_report.json" "$OUTPUT/raw_generation_report.json"
echo "single-stage redaction and merge complete: $OUTPUT"
