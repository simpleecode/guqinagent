#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20223393ljw/guqin-agent
ADAPTER=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016
INPUT="$ROOT/train/eval_inputs_v2_text_protocol/five_scores_latest_full/test_runtime.jsonl"
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/five_scores_latest_runtime_2gpu_20261020"
EXPECTED_SHA=7d1b835333ef0e1a697e50ad420c4c65850bd528dea7c4de24a6fe368b8e69ab

cd "$ROOT"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
printf '%s  %s\n' "$EXPECTED_SHA" "$INPUT" | sha256sum --check --status
python - "$ADAPTER" "$INPUT" <<'PY'
import json
import os
import sys
from collections import Counter
from pathlib import Path

adapter, source = map(Path, sys.argv[1:])
for filename in ('adapter_config.json', 'adapter_model.safetensors',
                 'train_results.json', 'trainer_state.json'):
    assert (adapter / filename).stat().st_size > 0, filename
state = json.loads((adapter / 'trainer_state.json').read_text())
assert float(state['epoch']) >= 2.99
assert state['global_step'] == state['max_steps']
assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 2
rows = [json.loads(line) for line in source.open() if line.strip()]
assert Counter(row['score_key'] for row in rows) == {
    'S1zt293d': 6, 'S35eJqgV': 66, 'SBFpPuMq': 6,
    'Sx4mbdGp': 15, 'SxhX3fgD': 4,
}
assert len(rows) == len({row['sample_id'] for row in rows}) == 97
assert all('reference' not in row and 'reference_plan' not in row['runtime_item']
           for row in rows)
print({'preflight': 'passed', 'phrases': len(rows),
       'final_epoch': state['epoch'], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']},
      flush=True)
PY
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$INPUT" --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B --adapter "$ADAPTER" \
  --vllm-env guqin-vllm-019 --workers 8 --server-count 2 --port 8910 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
