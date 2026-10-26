#!/usr/bin/env bash
set -euo pipefail
export GUQIN_TRAIN_GPUS=2
export GUQIN_TRAIN_CONFIG=train/configs/qwen35_9b_lora_two_stage_final_2gpu_bs12.yaml
export GUQIN_TRAIN_OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12
exec /bin/bash /home/20223393ljw/guqin-agent/train/scripts/run_two_stage_final_train_3gpu.sh
