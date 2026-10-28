#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent/.warning_ablation_runtime
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
export CUDA_VISIBLE_DEVICES=0
python -u - <<'PY'
import json, subprocess, time
stable = 0
while stable < 3:
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    stable = stable + 1 if free >= 20480 else 0
    print(json.dumps({'event': 'gpu_memory_wait', 'free_mib': free, 'stable_samples': stable}), flush=True)
    if stable < 3:
        time.sleep(20)
PY
python train/scripts/run_vllm_eval_queue.py \
  --input train/eval_outputs/base_two_stage/inputs.jsonl \
  --output-dir train/eval_outputs/base_two_stage/results \
  --warning-feedback on --workflow two_stage \
  --sampling-config train/configs/sampling_retry_schedule.json \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --vllm-env guqin-vllm-019 --server-count 1 --port 8993 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 3 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75
