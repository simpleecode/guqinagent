#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs_v3_two_stage/meihua_p0002_suffix_fingerchange_20261021
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
inputs=[json.loads(s) for s in (p/'inputs.jsonl').read_text().splitlines() if s.strip()]
frozen=[json.loads(s) for s in (p/'workers/predictions_00.jsonl').read_text().splitlines() if s.strip()]
assert [x['sample_id'] for x in inputs]==[f'S35eJqgV-p{i:04d}' for i in range(1,67)]
assert len(frozen)==1 and frozen[0]['sample_id']==inputs[0]['sample_id']
assert frozen[0]['input_sha256']==inputs[0]['input_sha256'] and frozen[0]['protocol_valid']
print('p0001 frozen; recomputing p0002–p0066 with current runtime.', flush=True)
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8898 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
