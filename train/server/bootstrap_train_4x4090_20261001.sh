#!/usr/bin/env bash
# Bootstrap and train the two-stage Qwen3.5-9B LoRA run on the 4x4090 host.
set -euo pipefail

ENV_PREFIX=/root/rivermind-data/envs/guqin-sft
WORK_ROOT=/root/rivermind-data/guqin-agent
MODEL_PATH=/root/rivermind-data/models/Qwen3.5-9B
CONFIG_PATH="$WORK_ROOT/train/configs/qwen35_9b_lora_4x4090_two_stage_v13_warningfix_20261001.yaml"
OUTPUT_PATH=/root/rivermind-data/guqin-agent/outputs/qwen35_9b_lora_4x4090_two_stage_v13_warningfix_20261001

mkdir -p "$(dirname "$MODEL_PATH")" "$OUTPUT_PATH"

if ! compgen -G "$MODEL_PATH/*.safetensors" >/dev/null; then
  echo "[$(date -Is)] downloading Qwen3.5-9B"
  "$ENV_PREFIX/bin/modelscope" download --model Qwen/Qwen3.5-9B --local_dir "$MODEL_PATH"
fi

echo "[$(date -Is)] validating 4-GPU runtime"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 4, f"expected 4 GPUs, got {torch.cuda.device_count()}"
print("CUDA", torch.version.cuda, "GPUs", [torch.cuda.get_device_name(i) for i in range(4)])
PY

echo "[$(date -Is)] starting training"
cd "$WORK_ROOT/LLaMA-Factory"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH="$ENV_PREFIX/bin:$PATH"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$ENV_PREFIX/bin/llamafactory-cli" train "$CONFIG_PATH" 2>&1 | tee "$OUTPUT_PATH/train_4epoch_20261001.log"
