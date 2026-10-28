#!/usr/bin/env bash
# Full sealed-test rollout for the production single-stage ReAct prompt plus
# the fixed complex-fingering handbook (macOS/Linux sibling of
# run_react_handbook_full.ps1). Resumable; runs the standard event-level
# evaluation automatically after the rollout finishes.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python}
MODEL=${MODEL:-glm-5.3}
MAX_ROUNDS=${MAX_ROUNDS:-8}
ATTEMPTS=${ATTEMPTS:-2}
SAMPLE_TEMPERATURE=${SAMPLE_TEMPERATURE:-0.25}
MIN_INTERVAL=${MIN_INTERVAL:-0.5}
RUN_NAME=${RUN_NAME:-react_handbook_full_test50}
INPUT_DIR=${INPUT_DIR:-$REPO/train/eval_inputs_v2_text_protocol/test50}
SOURCE_INPUT=$REPO/ABC_J/agent_training/two_stage_audited/inferred_all/inferred_trajectories_test.jsonl
SKIP_EVALUATION=${SKIP_EVALUATION:-0}

RUN_DIR=$REPO/baseline/gepa_guqin/runs/$RUN_NAME
PREDICTION_PATH=$RUN_DIR/predictions_test.jsonl
ROLLOUT_LOG=$RUN_DIR/rollout.log
EVALUATION_LOG=$RUN_DIR/evaluation.log
CONFIG_PATH=$RUN_DIR/launch_config.json
PID_PATH=$RUN_DIR/run.pid

export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO:$REPO/baseline/gepa/src${PYTHONPATH:+:$PYTHONPATH}"

for path in "$PYTHON" "$SOURCE_INPUT" "$REPO/.env"; do
    [ -e "$path" ] || { echo "Missing required file: $path" >&2; exit 1; }
done
for name in GLM_API_KEY GLM_BASE_URL GLM_MODEL; do
    grep -qE "^\s*$name=" "$REPO/.env" || { echo "Missing $name in .env" >&2; exit 1; }
done

mkdir -p "$RUN_DIR"

if [ -f "$PID_PATH" ] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "Run '$RUN_NAME' is already active as process $(cat "$PID_PATH")." >&2
    exit 1
fi
rm -f "$PID_PATH"

if [ ! -f "$INPUT_DIR/test_runtime.jsonl" ] || [ ! -f "$INPUT_DIR/evaluation_pairs_test.jsonl" ]; then
    echo "Exporting sealed public runtime input and private evaluation pairs..."
    "$PYTHON" -X utf8 -m baseline.gepa_guqin.export_eval_inputs \
        --source "$SOURCE_INPUT" --out-dir "$INPUT_DIR"
fi

CONFIG_JSON=$(cat <<EOF
{
  "schema_version": "react-handbook-launch-1.0",
  "run_name": "$RUN_NAME",
  "model": "$MODEL",
  "prompt": "production_seed",
  "handbook": true,
  "input": "$(realpath --relative-to="$REPO" "$INPUT_DIR")/test_runtime.jsonl",
  "reference": "$(realpath --relative-to="$REPO" "$INPUT_DIR")/evaluation_pairs_test.jsonl",
  "max_rounds": $MAX_ROUNDS,
  "attempts": $ATTEMPTS,
  "sample_temperature": $SAMPLE_TEMPERATURE,
  "min_interval": $MIN_INTERVAL
}
EOF
)
if [ -f "$CONFIG_PATH" ]; then
    for field in model prompt handbook input reference; do
        prior=$(python3 -c "import json;print(json.load(open('$CONFIG_PATH')).get('$field'))")
        current=$(printf '%s' "$CONFIG_JSON" | python3 -c "import json,sys;print(json.load(sys.stdin).get('$field'))")
        [ "$prior" = "$current" ] || { echo "Existing run uses a different '$field'; choose a new RUN_NAME." >&2; exit 1; }
    done
else
    printf '%s\n' "$CONFIG_JSON" > "$CONFIG_PATH"
fi

echo $$ > "$PID_PATH"
cd "$REPO"
trap 'rm -f "$PID_PATH"' EXIT

echo "Starting ReAct + Handbook full test: $RUN_NAME"
echo "Predictions: $PREDICTION_PATH"
echo "Live log: tail -f $ROLLOUT_LOG"
# Deliberately omit --prompt-json and --no-handbook: this selects the
# production seed prompt plus the fixed handbook.
"$PYTHON" -X utf8 -m baseline.gepa_guqin.run_final_eval \
    --input "$INPUT_DIR/test_runtime.jsonl" \
    --output "$PREDICTION_PATH" \
    --model "$MODEL" \
    --max-rounds "$MAX_ROUNDS" \
    --attempts "$ATTEMPTS" \
    --sample-temperature "$SAMPLE_TEMPERATURE" \
    --min-interval "$MIN_INTERVAL" \
    --resume 2>&1 | tee -a "$ROLLOUT_LOG"

if [ "$SKIP_EVALUATION" != "1" ]; then
    echo "Rollout complete; starting standard event-level evaluation."
    "$PYTHON" -X utf8 -m evaluation.run_eval \
        --pred "$PREDICTION_PATH" \
        --reference "$INPUT_DIR/evaluation_pairs_test.jsonl" \
        --experiment "$RUN_NAME" \
        --model "$MODEL+handbook" 2>&1 | tee -a "$EVALUATION_LOG"
fi
