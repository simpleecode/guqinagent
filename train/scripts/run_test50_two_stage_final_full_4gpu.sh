#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
MODEL=/home/20223393ljw/models/Qwen3.5-9B
ADAPTER=${GUQIN_EVAL_ADAPTER:-/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_3gpu_bs12}
export GUQIN_EVAL_GPUS=${GUQIN_EVAL_GPUS:-4}
INPUT="$ROOT/train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl"
OUTPUT="$ROOT/train/eval_outputs/two_stage/.work/test50"
EXPECTED_INPUT_SHA=250c19dba2dd2b39161148e5dcb7c29b428398a234a4a3dd237d19688a281269

cd "$ROOT"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
printf '%s  %s\n' "$EXPECTED_INPUT_SHA" "$INPUT" | sha256sum --check --status
test -s "$ADAPTER/adapter_config.json"
test -s "$ADAPTER/adapter_model.safetensors"
test -s "$ADAPTER/train_results.json"
python - "$ADAPTER/trainer_state.json" <<'PY'
import json
import os
import sys

state = json.load(open(sys.argv[1], encoding='utf-8'))
assert float(state['epoch']) >= 2.99, 'training did not finish 3 epochs'
assert state['global_step'] == state['max_steps'], 'training did not finish all steps'
assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == int(os.environ['GUQIN_EVAL_GPUS'])
PY
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
python train/scripts/run_vllm_eval_queue.py \
  --input "$INPUT" --output-dir "$OUTPUT" \
  --base-model "$MODEL" --adapter "$ADAPTER" \
  --vllm-env guqin-vllm-019 --workers "$((GUQIN_EVAL_GPUS * 4))" --server-count "$GUQIN_EVAL_GPUS" --port 8900 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
python train/scripts/visualize_completed_eval.py --publish --method two_stage \
  --inputs "$INPUT" --predictions "$OUTPUT/predictions.jsonl" --render
