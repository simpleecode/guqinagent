#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${GUQIN_CONDA_SH:-/home/20223393ljw/miniconda3/etc/profile.d/conda.sh}"
conda activate guqin-sft
DATA=train/data/guqin_direct_sft
CONFIG=train/configs/qwen35_9b_lora_direct_sft_1gpu_bs12.yaml
OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_direct_sft_1gpu_bs12
test ! -e "$OUTPUT"
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python train/scripts/verify_direct_sft.py --source-dir train/data/guqin_agent_sft_single_stage_final \
  --test-input .warning_ablation_runtime/train/eval_outputs/base_two_stage/inputs.jsonl \
  --modern-input .warning_ablation_runtime/train/eval_outputs/modern_comparison/inputs.jsonl
python train/scripts/prepare_direct_training_cache.py --config "$CONFIG"
python -u - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path
data = Path('train/data/guqin_direct_sft')
r = json.loads((data/'tokenized_manifest.json').read_text())
assert r['dataset_sha256'] == 'edba4abf12f1d2834bd079ea30f7ed1ef50238b2a631db6948712b18d102f7e9'
assert r['rows'] == r['complete_supervised_artifacts'] == 3976
assert r['template'] == 'qwen3_5' and r['cutoff_len'] == 8192
assert r['enable_thinking'] is False and r['train_on_prompt'] is False
gpu = os.environ['CUDA_VISIBLE_DEVICES']
assert gpu and ',' not in gpu
stable = 0
while stable < 3:
    used = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    stable = stable + 1 if used <= 512 else 0
    print(f'dedicated_gpu_wait gpu={gpu} used_mib={used} stable_samples={stable}', flush=True)
    if stable < 3:
        time.sleep(20)
PY
mkdir "$OUTPUT"
export FORCE_TORCHRUN=1 NPROC_PER_NODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train "$CONFIG" > "$OUTPUT/train.log" 2>&1
