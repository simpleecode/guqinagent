#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs/two_stage/.work/warning_suffix_retry
python - "$OUT" <<'PY'
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'selection_report.json').read_text())
assert hashlib.sha256((p/'inputs.jsonl').read_bytes()).hexdigest()==r['input_sha256']
for f,h in r['runtime_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --sampling-config train/configs/sampling_retry_schedule.json \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12 \
  --vllm-env guqin-vllm-019 --server-count "${SERVER_COUNT:-1}" --port 8920 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 3 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
