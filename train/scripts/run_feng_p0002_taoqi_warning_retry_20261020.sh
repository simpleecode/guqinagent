#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
OUT="$ROOT/train/eval_outputs_v3_two_stage/feng_p0002_taoqi_warning_retry_20261020"
cd "$ROOT"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(s) for s in (out / 'inputs.jsonl').read_text().splitlines() if s.strip()]
frozen = [json.loads(s) for s in (out / 'workers/predictions_00.jsonl').read_text().splitlines() if s.strip()]
assert [r['sample_id'] for r in rows] == ['S1zt293d-p0001', 'S1zt293d-p0002']
assert len(frozen) == 1 and frozen[0]['sample_id'] == rows[0]['sample_id']
assert frozen[0]['input_sha256'] == rows[0]['input_sha256']
assert frozen[0]['protocol_valid'] and frozen[0]['guqinizer']['ok']
assert len([d for d in __import__('os').environ['CUDA_VISIBLE_DEVICES'].split(',') if d]) == 1
print('p0001 frozen; p0002 full two-stage rerun', flush=True)
PY

exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8894 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
