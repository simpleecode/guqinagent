#!/usr/bin/env bash
# Rebuild both teacher stages for phrases containing a double-stopped 撮.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${TEACHER_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python}"
INPUT="$ROOT/ABC_J/agent_training/inferred_gqs_v12_tuningfix_20260914/inferred_trajectories_train.jsonl"
IDS="$ROOT/ABC_J/agent_training/cuo_fingering_audit_20260919/double_stopped_phrase_ids.txt"
RUN_ROOT="$ROOT/ABC_J/agent_training/cuo_double_stopped_rerun_20260919"
mkdir -p "$RUN_ROOT"
"$PYTHON" -c 'import anthropic' || {
  echo "Teacher Python lacks the anthropic package: $PYTHON" >&2
  exit 1
}

pids=()
for shard in 0 1 2 3 4 5 6; do
  out="$RUN_ROOT/raw_shard_$shard"
  log="$RUN_ROOT/raw_shard_$shard.log"
  if [[ -f "$out/generation_report.json" ]]; then
    echo "skip shard $shard: existing report $out/generation_report.json"
    continue
  fi
  # Ignore terminal hangup: desktop terminal commands otherwise clean up
  # ordinary background children when the invoking shell exits.
  nohup env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "$PYTHON" "$ROOT/ABC_J/scripts/generate_teacher_tool_trajectories.py" \
      --input "$INPUT" \
      --trajectory-id-file "$IDS" \
      --output-dir "$out" \
      --limit 100 \
      --include-guqinizer-no-op \
      --allow-private-reasoning-leakage \
      --max-attempts 4 \
      --min-interval 1 \
      --shard-count 7 \
      --shard-index "$shard" >"$log" 2>&1 < /dev/null &
  pids+=("$!")
  echo "started shard $shard (pid $!), log: $log"
done

# Keep this supervisor alive when run in a detached screen session.
wait "${pids[@]}"
