#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
OUT="$ROOT/train/eval_outputs_v3_two_stage/meihua_p0002_suffix_related_prompt_20261021"
ADAPTER=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016
cd "$ROOT"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019

python - "$OUT" "$ADAPTER" <<'PY'
import json
import os
import sys
from pathlib import Path

out, adapter = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in (out / 'inputs.jsonl').open() if line.strip()]
frozen = [json.loads(line) for line in
          (out / 'workers/predictions_00.jsonl').open() if line.strip()]
assert [row['sample_id'] for row in rows] == [f'S35eJqgV-p{i:04d}' for i in range(1, 67)]
assert [row['sample_id'] for row in frozen] == ['S35eJqgV-p0001']
assert frozen[0]['input_sha256'] == rows[0]['input_sha256']
assert frozen[0]['protocol_valid'] and frozen[0]['guqinizer']['ok']
assert len([gpu for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if gpu]) == 1
assert (adapter / 'adapter_model.safetensors').stat().st_size > 0
print('Preflight passed: p0001 frozen, p0002–p0066 to recompute.', flush=True)
PY

exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter "$ADAPTER" --vllm-env guqin-vllm-019 \
  --workers 1 --server-count 1 --port 8898 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
