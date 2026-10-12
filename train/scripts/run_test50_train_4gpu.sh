#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
cd "$ROOT"
python train/scripts/verify_test50_training_inputs.py
python -c 'import json,os; p=json.load(open("train/data/guqin_agent_sft_two_stage_test50/preflight_report.json")); assert p["rows"] == 7990 and not p["template_errors"] and p["cutoff_len"] == 8192 and p["truncation_rate"] <= p["max_truncation_rate"]; assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 4'
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train train/configs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012.yaml > "$OUTPUT/train.log" 2>&1
