#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs_v3_two_stage/meihua_p0004_suffix_openornament_20261021
python - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
inputs=[json.loads(s) for s in (p/'inputs.jsonl').read_text().splitlines() if s.strip()]
frozen=[json.loads(s) for s in (p/'workers/predictions_00.jsonl').read_text().splitlines() if s.strip()]
assert [x['sample_id'] for x in inputs]==[f'S35eJqgV-p{i:04d}' for i in range(1,67)]
assert [x['sample_id'] for x in frozen]==[x['sample_id'] for x in inputs[:3]]
assert all(x['input_sha256']==y['input_sha256'] and x['protocol_valid'] for x,y in zip(frozen,inputs))
old=Path('train/eval_outputs_v3_two_stage/meihua_p0035_suffix_inherited_string_20261021/workers/predictions_00.jsonl')
assert frozen==[json.loads(s) for s in old.read_text().splitlines() if s.strip()][:3]
files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
       'agents/ToolRuntime/warning_rendering.py','agents/ToolRuntime/warning_registry.py',
       'scripts/audit_jianpu_jianzi_pitch.py','agents/abc_to_jianzipu/teacher_trajectory.py']
(p/'runtime_manifest.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print('p0001–p0003 frozen; recomputing p0004–p0066.',flush=True)
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8898 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
