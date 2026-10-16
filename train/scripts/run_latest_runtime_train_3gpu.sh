#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
OUTPUT=${TRAIN_OUTPUT_DIR:-/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_latest_runtime_3gpu_bs12_20261015}
DATA=${TRAIN_DATA_DIR:-"$ROOT/train/data/guqin_agent_sft_two_stage_latest_runtime"}
CONFIG=${TRAIN_CONFIG:-train/configs/qwen35_9b_lora_latest_runtime_3gpu_bs12_20261015.yaml}
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
cd "$ROOT"
python - "$OUTPUT" <<'PY'
from pathlib import Path
import sys,time
p=Path(sys.argv[1])
if p.exists():
    # A model-loading failure left only the launch log, not trained weights.
    # Never overwrite a checkpoint or resume an unrelated adapter implicitly.
    assert all(f.is_file() and (f.name=='train.log' or f.name.startswith('train.launch_failed_')) for f in p.iterdir()), 'Output contains training artifacts; refusing overwrite'
    log=p/'train.log'
    if log.exists():log.rename(p/f'train.launch_failed_{time.time_ns()}.log')
PY
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python train/scripts/verify_test50_training_inputs.py --dataset-dir "$DATA" --expected-rows 7952
python - <<'PY'
import importlib.metadata as m
import os,torch
import liger_kernel
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==3
assert torch.cuda.is_available() and torch.cuda.device_count()==3
print({'environment':os.environ['CONDA_DEFAULT_ENV'],'torch':torch.__version__,
       'liger_kernel':m.version('liger-kernel'),'gpus':os.environ['CUDA_VISIBLE_DEVICES']},flush=True)
PY
if [[ -n "${VERIFIED_PREFLIGHT_DATA_SHA256:-}" ]]; then
python - "$DATA" "$VERIFIED_PREFLIGHT_DATA_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
h=hashlib.sha256()
with (p/'guqin_agent_train.jsonl').open('rb') as f:
    while chunk:=f.read(1024*1024):h.update(chunk)
assert h.hexdigest()==sys.argv[2], 'Cached preflight data hash changed'
r=json.loads((p/'preflight_report.json').read_text())
assert r['rows']==7952 and r['cutoff_len']==8192 and not r['template_errors']
assert r['truncation_rate'] <= r['max_truncation_rate']
if (p/'tokenized_manifest.json').exists():
    cached=json.loads((p/'tokenized_manifest.json').read_text())
    assert cached['dataset_sha256']==h.hexdigest() and cached['rows']==7952
    assert cached['cutoff_len']==8192 and cached['template']=='qwen3_5'
    assert cached['train_on_prompt'] is False and cached['mask_history'] is False
    assert Path(cached['tokenized_path']).is_dir()
print('Reusing successful preflight for verified immutable dataset '+h.hexdigest(),flush=True)
PY
else
python train/scripts/preflight_sft.py --input "$DATA/guqin_agent_train.jsonl" \
  --model-path /home/20223393ljw/models/Qwen3.5-9B --cutoff-len 8192 \
  --report "$DATA/preflight_report.json"
fi
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train "$CONFIG" > "$OUTPUT/train.log" 2>&1
