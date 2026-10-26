#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent/.warning_ablation_runtime
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
python train/scripts/prepare_warning_ablation.py
python -u - <<'PY'
import os,subprocess,time,json
devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');stable=0
while stable<3:
    free=[int(subprocess.check_output(['nvidia-smi','-i',d,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for d in devices]
    stable=stable+1 if min(free)>=20480 else 0
    print(json.dumps({'event':'gpu_memory_wait','free_mib':free,'stable_samples':stable}),flush=True)
    if stable<3:time.sleep(20)
PY
for arm in on off; do
  python train/scripts/run_vllm_eval_queue.py \
    --input train/eval_outputs/warning_ablation/pilot/inputs.jsonl \
    --output-dir "train/eval_outputs/warning_ablation/pilot/$arm" \
    --warning-feedback "$arm" --workflow two_stage \
    --sampling-config train/configs/sampling_retry_schedule.json \
    --base-model /home/20223393ljw/models/Qwen3.5-9B \
    --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12 \
    --vllm-env guqin-vllm-019 --server-count 1 --port 8990 \
    --max-new-tokens 8192 --max-rounds 8 --attempts 3 --max-guqinizer-passes 3 \
    --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75
done
