#!/usr/bin/env bash
# Run both sealed evaluation sets without altering the existing Agent experiments.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${GUQIN_EVAL_PYTHON:-/home/20223393ljw/miniconda3/envs/guqin-vllm-019/bin/python}"
ADAPTER=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_direct_sft_1gpu_bs12
OUTPUT=train/eval_outputs/direct_sft
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/.coordinator.lock"
flock -n 9 || { echo 'Direct evaluation coordinator already running'; exit 1; }
"$PYTHON" -u - <<'PY'
import hashlib, json
from pathlib import Path
expected = {
    '/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_direct_sft_1gpu_bs12/adapter_model.safetensors':
        '30c0e8fcd8fb7387d86f66677f8596c24327a693eced4f5ac8131dfcb27d0913',
    'train/eval_outputs/base_two_stage/inputs.jsonl':
        '7670e320abe0900a35738c90f7fed2517feffd2ce04d26e51a3175e135301fa4',
    'train/eval_outputs/modern_comparison/inputs.jsonl':
        '8c37783873f22f07f397e96ebe07d7009e086c69e7687f57a03a45d3c2079400',
}
for name, sha in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
for scope in ('test48', 'modern'):
    assert not (Path('train/eval_outputs/direct_sft') / scope / 'direct_config.json').exists(), (
        'Existing evaluation: inspect before explicitly resuming', scope)
print(json.dumps({'event': 'direct_preflight_verified', 'sha256': expected}), flush=True)
PY
for scope in test48 modern; do
    input=train/eval_outputs/base_two_stage/inputs.jsonl
    if [[ "$scope" == modern ]]; then input=train/eval_outputs/modern_comparison/inputs.jsonl; fi
    echo "direct_scope_start $scope $(date -Iseconds)"
    "$PYTHON" -u train/scripts/run_direct_vllm_eval.py \
        --input "$input" --output-dir "$OUTPUT/$scope" \
        --base-model /home/20223393ljw/models/Qwen3.5-9B \
        --adapter "$ADAPTER" --workers 8 --port 8999
    echo "direct_scope_finished $scope $(date -Iseconds)"
done
echo "direct_all_finished $(date -Iseconds)"
