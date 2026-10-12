#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
MODEL=/home/20223393ljw/models/Qwen3.5-9B
TRAIN_OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012
ADAPTER="$TRAIN_OUTPUT"
INPUT="$ROOT/train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl"
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/two_stage_test50_3epoch"

test -s "$ADAPTER/adapter_config.json"
test -s "$TRAIN_OUTPUT/train_results.json"
test -s "$INPUT"
mkdir -p "$OUTPUT"

source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019

cd "$ROOT"
python train/scripts/verify_test50_training_inputs.py
# Jobber exports CUDA_VISIBLE_DEVICES.
# Preserve that allocation instead of hard-coding device IDs.
python train/scripts/run_vllm_eval_queue.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT" \
  --base-model "$MODEL" \
  --adapter "$ADAPTER" \
  --vllm-env guqin-vllm-019 \
  --workers 16 \
  --server-count 2 \
  --port 8310 \
  --max-new-tokens 8192 \
  --max-rounds 8 \
  --attempts 2 \
  --max-guqinizer-passes 3 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
