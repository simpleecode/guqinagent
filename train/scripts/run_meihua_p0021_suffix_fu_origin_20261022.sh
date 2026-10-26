#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
OUT=train/eval_outputs_v3_two_stage/meihua_p0021_suffix_fu_origin_20261022
python - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
read=lambda f:[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
inputs=read(p/'inputs.jsonl')
frozen=read(p/'workers/predictions_00.jsonl')
assert [r['sample_id'] for r in inputs]==[f'S35eJqgV-p{i:04d}' for i in range(1,67)]
assert len(frozen)==20
assert all(r['sample_id']==s['sample_id'] and r['input_sha256']==s['input_sha256']
           and r['protocol_valid'] for r,s in zip(frozen,inputs))
assert frozen==read(Path('train/eval_outputs_v3_two_stage/meihua_p0065_suffix_no_early_stop_20261022/predictions.jsonl'))[:20]
assert 'if repeated_submission:' not in Path('train/scripts/eval_two_stage_score.py').read_text()
files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
       'agents/ToolRuntime/warning_rendering.py','agents/ToolRuntime/warning_registry.py',
       'scripts/audit_jianpu_jianzi_pitch.py','agents/abc_to_jianzipu/teacher_trajectory.py',
       'ABC_J/scripts/generate_teacher_tool_trajectories.py','train/scripts/eval_two_stage_score.py']
manifest={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
assert manifest==json.loads((p/'expected_runtime_manifest.json').read_text()), 'Runtime changed while queued'
(p/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen p0001–p0020; rerunning both stages of p0021–p0066 with fu-origin audit.',flush=True)
PY
exec python train/scripts/run_vllm_eval_queue.py \
  --input "$OUT/inputs.jsonl" --output-dir "$OUT" \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016 \
  --vllm-env guqin-vllm-019 --workers 1 --server-count 1 --port 8898 \
  --max-new-tokens 8192 --max-rounds 8 --attempts 2 --max-guqinizer-passes 3 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.75 \
  > "$OUT/eval.log" 2>&1
