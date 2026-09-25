#!/usr/bin/env bash
# Periodically pull a password-authenticated remote evaluation JSONL and
# refresh the existing running-evaluation HTML viewer.  The password is read
# only from EVAL_SSH_PASSWORD; it is deliberately not stored in this script.
set -uo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 HOST PORT REMOTE_JSONL LOCAL_JSONL HTML SCORE INTERVAL_SECONDS REPOSITORY_ROOT" >&2
  exit 2
fi

host=$1
port=$2
remote_jsonl=$3
local_jsonl=$4
html=$5
score=$6
interval=$7
root=$8
# A training-fit score may not exist in the default held-out test input.
# Allow the caller to select the exact public input used by the remote run.
input_path=${EVAL_INPUT_PATH:-"$root/train/eval_inputs_v2_text_protocol/test.jsonl"}
references_path=${EVAL_REFERENCES_PATH:-"$root/ABC_J/agent_training/reference_trajectories_test.jsonl"}

: "${EVAL_SSH_PASSWORD:?set EVAL_SSH_PASSWORD in the invoking environment}"
mkdir -p "$(dirname "$local_jsonl")" "$(dirname "$html")"

pull_once() {
  local partial="${local_jsonl}.partial"
  EVAL_SYNC_HOST="$host" EVAL_SYNC_PORT="$port" \
  EVAL_SYNC_REMOTE="$remote_jsonl" EVAL_SYNC_LOCAL="$partial" expect -c '
    set timeout 120
    set password $env(EVAL_SSH_PASSWORD)
    set host $env(EVAL_SYNC_HOST)
    set port $env(EVAL_SYNC_PORT)
    set remote $env(EVAL_SYNC_REMOTE)
    set local $env(EVAL_SYNC_LOCAL)
    spawn scp -P $port -o PreferredAuthentications=password -o PubkeyAuthentication=no "$host:$remote" $local
    expect {
      "password:" {send -- "$password\r"; exp_continue}
      eof {}
    }
  ' && mv -f "$partial" "$local_jsonl"
}

while true; do
  if pull_once; then
    python "$root/ABC_J/scripts/visualize_running_eval_score.py" \
      --score "$score" \
      --input "$input_path" \
      --references "$references_path" \
      --pulled "$local_jsonl" \
      --output "$html" \
      --focus-sample "${score}-p0001" \
      --skip-pull || echo "$(date '+%F %T') render failed" >&2
  else
    echo "$(date '+%F %T') pull failed; retrying" >&2
  fi
  sleep "$interval"
done
