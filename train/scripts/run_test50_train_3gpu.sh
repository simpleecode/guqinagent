#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_3gpu_bs12_20261012
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
cd "$ROOT"
test -s train/data/guqin_agent_sft_two_stage_test50/sft_export_validation_report.json
test -s train/eval_inputs_v2_text_protocol/test50/export_manifest.json
test -n "${CUDA_VISIBLE_DEVICES:-}"
python train/scripts/verify_test50_training_inputs.py
python -c 'import json; p=json.load(open("train/data/guqin_agent_sft_two_stage_test50/preflight_report.json")); assert p["rows"] == 7990 and not p["template_errors"] and p["cutoff_len"] == 8192 and p["truncation_rate"] <= p["max_truncation_rate"]'
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train train/configs/qwen35_9b_lora_two_stage_test50_3gpu_bs12_20261012.yaml > "$OUTPUT/train.log" 2>&1
