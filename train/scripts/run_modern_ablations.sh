#!/usr/bin/env bash
set -euo pipefail
runtime=${GUQIN_RUNTIME:-/home/20223393ljw/guqin-agent/.warning_ablation_runtime}
cd "$runtime"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
input=train/eval_outputs/modern_comparison/inputs.jsonl
printf '%s  %s\n' 8c37783873f22f07f397e96ebe07d7009e086c69e7687f57a03a45d3c2079400 "$input" | sha256sum -c
printf '%s  %s\n' ecdee4141e31cabad373a7dba0fbcddcf0a03d15ff11f46bb41ca02ec510d0ad /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12/adapter_model.safetensors | sha256sum -c
printf '%s  %s\n' 8447be00f62ad6a81c605b5d1cb59e7b7149582b7caddec987e33ba9ee57ac29 /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_fixed_reference_ablation_1gpu_bs12/adapter_model.safetensors | sha256sum -c
# Refuse duplicate launches. Resume must be deliberate after log inspection.
mkdir train/eval_outputs/modern_comparison/.ablations_launch
for method in two_stage_off fixed_workflow; do
  python -u - <<'PY'
import os, subprocess, time
gpu = os.environ['CUDA_VISIBLE_DEVICES']
assert gpu == '0'
stable = 0
while stable < 3:
    free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    stable = stable + 1 if free >= 20480 else 0
    print(f'gpu_memory_wait free_mib={free} stable_samples={stable}', flush=True)
    if stable < 3:
        time.sleep(20)
PY
  if [[ "$method" == two_stage_off ]]; then
    adapter=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12
    workflow=two_stage; feedback=off; rounds=8
  else
    adapter=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_fixed_reference_ablation_1gpu_bs12
    workflow=single_stage; feedback=on; rounds=16
  fi
  python train/scripts/run_vllm_eval_queue.py \
    --input "$input" --output-dir "train/eval_outputs/modern_comparison/$method" \
    --warning-feedback "$feedback" --workflow "$workflow" \
    --sampling-config train/configs/sampling_retry_schedule.json \
    --base-model /home/20223393ljw/models/Qwen3.5-9B --adapter "$adapter" \
    --vllm-env guqin-vllm-019 --server-count 1 --workers 8 --port 8998 \
    --max-new-tokens 8192 --max-rounds "$rounds" --attempts 3 --max-guqinizer-passes 3 \
    --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75
done
