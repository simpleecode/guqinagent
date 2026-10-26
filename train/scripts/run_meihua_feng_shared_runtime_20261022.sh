#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs/two_stage/.work/inference
python - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
read=lambda f:[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
inputs=read(p/'inputs.jsonl'); frozen=read(p/'workers/predictions_00.jsonl')
assert len(inputs)==len({r['sample_id'] for r in inputs})==72
assert {r['sample_id'] for r in inputs}==({f'S35eJqgV-p{i:04d}' for i in range(1,67)}|{f'S1zt293d-p{i:04d}' for i in range(1,7)})
source={r['sample_id']:r for r in inputs}
assert len(frozen)==20
assert all(r['input_sha256']==source[r['sample_id']]['input_sha256'] and r['protocol_valid'] for r in frozen)
assert frozen==read(Path('train/eval_outputs/two_stage/S35eJqgV/predictions.jsonl'))[:20]
files=list(json.loads((p/'expected_runtime_manifest.json').read_text()))
manifest={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
assert manifest==json.loads((p/'expected_runtime_manifest.json').read_text()), 'Runtime changed while queued'
(p/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Single shared vLLM: Meihua p0021–p0066 + all 6 Feng phrases; 20 Meihua phrases frozen.',flush=True)
PY
python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8898 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
python train/scripts/visualize_completed_eval.py --publish --method two_stage \
  --inputs "$OUT/inputs.jsonl" --predictions "$OUT/predictions.jsonl" --render
