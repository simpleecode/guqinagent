#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
DATA="$ROOT/train/eval_inputs_v2_text_protocol/Sx4mbdGp_p0013_suffix"
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/Sx4mbdGp_p0013_suffix_repeatfix_20261013"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
python - <<'PY'
import json
from pathlib import Path
base=Path('train/eval_inputs_v2_text_protocol/Sx4mbdGp_p0013_suffix')
rows=[json.loads(l) for l in (base/'test_runtime.jsonl').open()]
prefix=[json.loads(l) for l in (base/'prefix_predictions.jsonl').open()]
assert len(rows)==15 and {r['score_key'] for r in rows}=={'Sx4mbdGp'}
assert {r['phrase_id'] for r in prefix}=={f'p{i:04d}' for i in range(1,13)}
hashes={r['sample_id']:r['input_sha256'] for r in rows}
assert all(r['protocol_valid'] and r['input_sha256']==hashes[r['sample_id']] for r in prefix)
print('Resume preflight passed: 12 frozen prefix phrases; regenerate p0013-p0015.',flush=True)
PY
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT/workers"
cp "$DATA/prefix_predictions.jsonl" "$OUTPUT/workers/predictions_00.jsonl"
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$DATA/test_runtime.jsonl" --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012 \
  --vllm-env guqin-vllm-019 --workers 16 --server-count 1 --port 8660 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
