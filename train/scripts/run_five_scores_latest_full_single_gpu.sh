#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs_v3_two_stage/five_scores_latest_runtime_full_1gpu_20261018
test ! -e "$OUT"
mkdir -p "$OUT"
exec python train/scripts/run_vllm_eval_queue.py \
  --input train/eval_inputs_v2_text_protocol/five_scores_latest_full/test_runtime.jsonl \
  --output-dir "$OUT" --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 8 --server-count 1 --port 8890 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
