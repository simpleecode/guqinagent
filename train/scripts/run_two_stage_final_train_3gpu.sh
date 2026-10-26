#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
DATA="$ROOT/train/data/guqin_agent_sft_two_stage_final"
OUTPUT=${GUQIN_TRAIN_OUTPUT:-/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_3gpu_bs12}
CONFIG=${GUQIN_TRAIN_CONFIG:-train/configs/qwen35_9b_lora_two_stage_final_3gpu_bs12.yaml}
GPU_COUNT=${GUQIN_TRAIN_GPUS:-3}
export GUQIN_TRAIN_GPUS="$GPU_COUNT"
EXPECTED_SHA=6ea8329df61ade844389611d76ba5e3215deba19f9a9aa6e7e0ae1d8c8d150b6

cd "$ROOT"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
test ! -e "$OUTPUT"
printf '%s  %s\n' "$EXPECTED_SHA" "$DATA/guqin_agent_train.jsonl" | sha256sum --check --status
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python train/scripts/verify_test50_training_inputs.py --dataset-dir "$DATA" --expected-rows 7952
python - "$DATA" <<'PY'
import json
import os
import sys
from pathlib import Path

data = Path(sys.argv[1])
preflight = json.loads((data / 'preflight_report.json').read_text())
cached = json.loads((data / 'tokenized_manifest.json').read_text())
assert preflight['rows'] == 7952 and preflight['cutoff_len'] == 8192
assert not preflight['template_errors']
assert preflight['truncation_rate'] <= preflight['max_truncation_rate']
assert cached['dataset_sha256'] == '6ea8329df61ade844389611d76ba5e3215deba19f9a9aa6e7e0ae1d8c8d150b6'
assert cached['rows'] == 7952 and cached['cutoff_len'] == 8192
assert cached['template'] == 'qwen3_5'
assert cached['train_on_prompt'] is False and cached['mask_history'] is False
assert Path(cached['tokenized_path']).is_dir()
assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == int(os.environ['GUQIN_TRAIN_GPUS'])
PY
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE="$GPU_COUNT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train "$CONFIG" > "$OUTPUT/train.log" 2>&1
