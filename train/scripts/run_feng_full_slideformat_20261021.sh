#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs_v3_two_stage/feng_full_slideformat_20261021
python - "$OUT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[json.loads(s) for s in (p/'inputs.jsonl').read_text().splitlines() if s.strip()]
assert [x['sample_id'] for x in rows]==[f'S1zt293d-p{i:04d}' for i in range(1,7)]
assert not (p/'workers').exists() and not (p/'predictions.jsonl').exists()
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
       'agents/ToolRuntime/warning_rendering.py','agents/ToolRuntime/warning_registry.py',
       'scripts/audit_jianpu_jianzi_pitch.py','agents/abc_to_jianzipu/teacher_trajectory.py']
(p/'runtime_manifest.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
print('All 6 phrases: full two-stage rerun with slide-attack format warning.',flush=True)
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8895 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
