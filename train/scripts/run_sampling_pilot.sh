#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent/.sampling_pilot_runtime
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
for arm in greedy precision; do
  python train/scripts/run_vllm_eval_queue.py \
    --input train/eval_outputs/sampling_pilot/inputs.jsonl \
    --output-dir "train/eval_outputs/sampling_pilot/$arm" \
    --sampling-config "train/configs/sampling_$arm.json" \
    --base-model /home/20223393ljw/models/Qwen3.5-9B \
    --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12 \
    --vllm-env guqin-vllm-019 --workers 4 --server-count 1 --port 8910 \
    --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
    --max-model-len 16384 --max-num-seqs 4 --gpu-memory-utilization 0.75
done
