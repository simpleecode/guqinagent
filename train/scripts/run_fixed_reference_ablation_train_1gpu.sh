#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
DATA=train/data/guqin_agent_sft_fixed_reference_ablation
CONFIG=train/configs/qwen35_9b_lora_fixed_reference_ablation_1gpu_bs12.yaml
OUTPUT=/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_fixed_reference_ablation_1gpu_bs12
test ! -e "$OUTPUT"
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python - <<'PY'
import csv, hashlib, json
from pathlib import Path
p = Path('train/data/guqin_agent_sft_fixed_reference_ablation')
m = json.loads((p/'dataset_manifest.json').read_text())
for name, key in [('guqin_agent_train.jsonl','dataset_sha256'), ('guqin_agent_train_metadata.jsonl','metadata_sha256')]:
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == m[key]
assert m['dataset_sha256'] == '72f664d51ca2aecb62e792d20043f8bfbefe494c64d10c1a4b6f0e68c81ffc6a'
rows = [json.loads(line) for line in (p/'guqin_agent_train_metadata.jsonl').open() if line.strip()]
assert len(rows) == len({r['source_sample_id'] for r in rows}) == 3976
assert all(r['agent_stage'] == 'single_stage' for r in rows)
train = {r['score_key'] for r in rows}
test_path = Path('.warning_ablation_runtime/train/eval_outputs/base_two_stage/inputs.jsonl')
assert hashlib.sha256(test_path.read_bytes()).hexdigest() == '7670e320abe0900a35738c90f7fed2517feffd2ce04d26e51a3175e135301fa4'
test = {json.loads(line)['score_key'] for line in test_path.open() if line.strip()}
with Path('ABC_J/results/agent_dataset_splits.csv').open(encoding='utf-8-sig') as stream:
    splits = {r['score_key']: r for r in csv.DictReader(stream)}
assert len(train) == 153 and len(test) == 48 and not train & test
assert all(splits[k]['split'] == 'train' for k in train)
assert all(splits[k]['split'] == 'test' for k in test)
assert not {splits[k]['leakage_group_id'] for k in train} & {splits[k]['leakage_group_id'] for k in test}
PY
if [[ ! -f "$DATA/tokenized_manifest.json" ]]; then
  python train/scripts/prepare_training_tokenized_cache.py --config "$CONFIG" --expected-rows 3976
fi
python - <<'PY'
import json, os
from pathlib import Path
r = json.loads(Path('train/data/guqin_agent_sft_fixed_reference_ablation/tokenized_manifest.json').read_text())
assert r['dataset_sha256'] == '72f664d51ca2aecb62e792d20043f8bfbefe494c64d10c1a4b6f0e68c81ffc6a'
assert r['rows'] == 3976 and r['cutoff_len'] == 8192 and r['template'] == 'qwen3_5'
assert r['train_on_prompt'] is False and r['mask_history'] is False
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
PY
mkdir -p "$OUTPUT"
export FORCE_TORCHRUN=1 NPROC_PER_NODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec llamafactory-cli train "$CONFIG" > "$OUTPUT/train.log" 2>&1
