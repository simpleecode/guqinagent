#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent/.warning_ablation_runtime
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
printf '%s  %s\n' 7670e320abe0900a35738c90f7fed2517feffd2ce04d26e51a3175e135301fa4 train/eval_outputs/base_two_stage/inputs.jsonl | sha256sum -c
python -u - <<'PY'
import json, os, subprocess, time
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices) == 2
stable = 0
while stable < 3:
    free = [int(subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) for device in devices]
    stable = stable + 1 if min(free) >= 20480 else 0
    print(json.dumps({'event': 'gpu_memory_wait', 'free_mib': free, 'stable_samples': stable}), flush=True)
    if stable < 3:
        time.sleep(20)
PY
python train/scripts/run_vllm_eval_queue.py \
  --input train/eval_outputs/base_two_stage/inputs.jsonl \
  --output-dir train/eval_outputs/single_stage_final/results \
  --warning-feedback on --workflow single_stage \
  --sampling-config train/configs/sampling_retry_schedule.json \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_single_stage_final_1gpu_bs12 \
  --vllm-env guqin-vllm-019 --server-count 2 --workers 16 --port 8994 \
  --max-new-tokens 8192 --max-rounds 16 --attempts 3 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75
