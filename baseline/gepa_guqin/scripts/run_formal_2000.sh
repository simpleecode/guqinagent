#!/bin/zsh
# Formal GEPA run: 2000 metric-call budget on the full optimization split.
# Review pilot results (runs/pilot_20261007/api_stats.json) before launching.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

PY=.venv-gepa/bin/python
DATA=baseline/gepa_guqin/data/gepa_split_v1
RUN=baseline/gepa_guqin/runs/opt_v1

# 1. optimization split (skip if the manifest already exists)
if [[ ! -f "$DATA/manifest.json" ]]; then
  $PY -m baseline.gepa_guqin.data --out-dir "$DATA" --phrases 600
fi

# 2. GEPA evolution (resumable: rerun this script to continue; touch
#    $RUN/gepa.stop to stop gracefully)
$PY -m baseline.gepa_guqin.run_optimize \
  --data-dir "$DATA" \
  --run-dir "$RUN" \
  --max-metric-calls 2000 \
  --reflection-minibatch-size 6 \
  --min-interval 0.3

# 3. sealed-test prediction + standard reporting (run manually when 2 finishes)
echo "next:"
echo "  $PY -m baseline.gepa_guqin.run_final_eval --prompt-json $RUN/best_candidate.json --output $RUN/predictions_test.jsonl --resume"
echo "  $PY -m evaluation.run_eval --pred $RUN/predictions_test.jsonl --reference train/eval_inputs_v2_text_protocol/renderfix_20261007/evaluation_pairs_test.jsonl --experiment gepa_single_stage_v1 --model glm-5.3+gepa"
