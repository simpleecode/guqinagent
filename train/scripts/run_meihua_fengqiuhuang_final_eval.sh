#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/20223393ljw/guqin-agent
ADAPTER=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_related_context_3gpu_bs12_03d06bae
INPUT="$ROOT/train/eval_inputs_v2_text_protocol/meihua_fengqiuhuang_latest"
OUTPUT="$ROOT/train/eval_outputs_v3_two_stage/related_context_final_meihua_fengqiuhuang_2gpu"
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
cd "$ROOT"
python - "$ADAPTER" "$INPUT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
from collections import Counter
from agents.abc_to_jianzipu.reference_parser import AUDIT
adapter,base=map(Path,sys.argv[1:])
for filename in ('adapter_config.json','adapter_model.safetensors','train_results.json','trainer_state.json'):
    assert (adapter/filename).stat().st_size>0,filename
state=json.loads((adapter/'trainer_state.json').read_text())
assert float(state['epoch'])>=2.99 and state['global_step']==state['max_steps'], 'Final training not complete'
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
manifest=json.loads((base/'manifest.json').read_text())
path=base/'test_runtime.jsonl'
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.name]['sha256']
rows=[json.loads(l) for l in path.open() if l.strip()]
assert Counter(r['score_key'] for r in rows)=={'S35eJqgV':66,'S1zt293d':6}
for row in rows:
    item=row['runtime_item']
    assert 'reference' not in row and 'reference_plan' not in item
    assert item['input']['normalized_tuning']['open_midi']==AUDIT.parse_open_midi(item['input']['metadata'])
print(json.dumps({'preflight':'passed','phrases':72,'final_epoch':state['epoch'],
                  'final_step':state['global_step'],'gpus':os.environ['CUDA_VISIBLE_DEVICES']},ensure_ascii=False),flush=True)
PY
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$INPUT/test_runtime.jsonl" --output-dir "$OUTPUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B --adapter "$ADAPTER" \
  --vllm-env guqin-vllm-019 --workers 8 --server-count 2 --port 8760 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  > "$OUTPUT/eval.log" 2>&1
