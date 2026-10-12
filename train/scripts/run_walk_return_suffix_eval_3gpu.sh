#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
if test -s "$ROOT/train/eval_inputs_v2_text_protocol/test50/export_manifest.json"; then
  echo 'Retired old-split suffix evaluation; use the verified test50 evaluation launcher.' >&2
  exit 2
fi
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/harmonicrestfix_checkpoint2574_walkreturn_repeatanchor_suffix_3gpu_20261011"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
python train/scripts/run_vllm_eval_queue.py \
  --input "$OUTPUT/input.jsonl" --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_harmonicrestfix_3x3090_bs12_20261010/checkpoint-2574 \
  --vllm-env guqin-vllm-019 --workers 16 --server-count 3 --port 8510 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval_3gpu.log" 2>&1
