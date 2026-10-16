#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/SzV4KbB5_p0005_suffixstop_retry_20261014"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
test ! -e "$OUTPUT"
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
output = Path(sys.argv[1])
rows = [json.loads(l) for l in Path('train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl').open()]
rows = sorted((r for r in rows if r['score_key'] == 'SzV4KbB5' and r['phrase_id'] <= 'p0005'), key=lambda r:r['phrase_id'])
prior = Path('train/eval_outputs_v3_two_stage/two_stage_test50_four_scores_harmonicrules_full_4gpu_20261014/predictions.jsonl')
prefix = sorted((r for l in prior.open() if (r:=json.loads(l))['score_key']=='SzV4KbB5' and r['phrase_id'] < 'p0005'), key=lambda r:r['phrase_id'])
assert len(rows)==5 and len(prefix)==4
hashes={r['sample_id']:r['input_sha256'] for r in rows}
assert all(r['protocol_valid'] and r['input_sha256']==hashes[r['sample_id']] for r in prefix)
(output/'workers').mkdir(parents=True)
for path, records in [(output/'test_runtime.jsonl',rows),(output/'workers/predictions_00.jsonl',prefix)]:
    with path.open('w') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('Preflight passed: freeze p0001–p0004; rerun p0005 Base + Guqinizer.', flush=True)
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUTPUT/test_runtime.jsonl" --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012 \
  --vllm-env guqin-vllm-019 --workers 8 --server-count 1 --port 8690 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
