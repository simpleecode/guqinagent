#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
OUTPUT="${EVAL_OUTPUT_DIR:-$ROOT/train/eval_outputs_v3_two_stage/SzV4KbB5_p0001_samepitch_retry_r2_20261014}"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
exec python train/scripts/run_vllm_eval_queue.py \
  --input train/eval_inputs_v2_text_protocol/SzV4KbB5_p0001_samepitch_retry/test_runtime.jsonl --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012 \
  --vllm-env guqin-vllm-019 --workers 8 --server-count 1 --port 8690 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
