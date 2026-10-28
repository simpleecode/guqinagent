#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
DATA=train/data/guqin_agent_sft_single_stage_final
CONFIG=train/configs/qwen35_9b_lora_single_stage_final_1gpu_bs12.yaml
OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_single_stage_final_1gpu_bs12
test ! -e "$OUTPUT"
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python train/scripts/verify_single_stage_training_inputs.py
if [[ ! -f "$DATA/tokenized_manifest.json" ]]; then
  python train/scripts/prepare_training_tokenized_cache.py --config "$CONFIG" --expected-rows 3976
fi
python - <<'PY'
import json,os
from pathlib import Path
p=Path('train/data/guqin_agent_sft_single_stage_final')
r=json.loads((p/'tokenized_manifest.json').read_text())
assert r['dataset_sha256']=='984779ea61a9c582328518bf8d4e023826bf31b8c705f927c1dfbd9d97b00f4a'
assert r['rows']==3976 and r['cutoff_len']==8192 and r['template']=='qwen3_5'
assert r['train_on_prompt'] is False and r['mask_history'] is False
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
PY
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train "$CONFIG" > "$OUTPUT/train.log" 2>&1
