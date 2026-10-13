#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
MODEL=/home/20223393ljw/models/Qwen3.5-9B
ADAPTER=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_test50_4gpu_bs12_20261012
INPUT_DIR="$ROOT/train/eval_inputs_v2_text_protocol/test50_structurefix_four_scores"
OUTPUT="${EVAL_OUTPUT_DIR:-$ROOT/train/eval_outputs_v3_two_stage/two_stage_test50_structurefix_four_scores_20261013}"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
test -s "$ADAPTER/adapter_config.json"
test -s "$ADAPTER/adapter_model.safetensors"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
from agents.abc_to_jianzipu.reference_parser import AUDIT
from agents.abc_to_jianzipu.score_structure import is_structural_note
base=Path('train/eval_inputs_v2_text_protocol/test50_structurefix_four_scores')
manifest=json.loads((base/'manifest.json').read_text())
assert hashlib.sha256((base/'test_runtime.jsonl').read_bytes()).hexdigest()==manifest['sha256']
rows=[json.loads(line) for line in (base/'test_runtime.jsonl').open() if line.strip()]
assert {r['score_key'] for r in rows}=={'Sx4mbdGp','SzV4KbB5','SBFpPuMq','Sc9dDref'}
assert len(rows)==manifest['phrases']==len({r['sample_id'] for r in rows})
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==4
for row in rows:
    data=row['runtime_item']['input']
    assert data['normalized_tuning']['open_midi']==AUDIT.parse_open_midi(data['metadata'])
    for note in data['notes_without_jianzi']:
        if is_structural_note(note):
            assert note['event_index'] is None and not note['abc'].startswith('z')
print(json.dumps({'preflight':'passed','phrases':len(rows),'gpus':os.environ['CUDA_VISIBLE_DEVICES']},ensure_ascii=False),flush=True)
PY
# Never append new inference to an older partial run.
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$INPUT_DIR/test_runtime.jsonl" --output-dir "$OUTPUT" \
  --base-model "$MODEL" --adapter "$ADAPTER" \
  --vllm-env guqin-vllm-019 --workers 16 --server-count 4 --port 8650 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
