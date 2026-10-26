#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-vllm-019
exec python - <<'PY'
import hashlib,json,os,sys,subprocess,time
from pathlib import Path
out=Path(os.environ.get('REFINEMENT_DIR','train/eval_outputs/two_stage/.work/hint_refinement'))
r=json.loads((out/'selection_report.json').read_text())
for name,digest in r['sha256'].items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
for name,digest in r['runtime_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
command=[sys.executable,'train/scripts/run_vllm_eval_queue.py',
 '--input',str(out/'inputs.jsonl'),'--output-dir',str(out),
 '--sampling-config','train/configs/sampling_retry_schedule.json',
 '--base-model','/home/20223393ljw/models/Qwen3.5-9B',
 '--adapter','/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12',
 '--vllm-env','guqin-vllm-019','--server-count',os.environ.get('SERVER_COUNT','2'),'--port',os.environ.get('REFINEMENT_PORT','8940'),
 '--max-new-tokens','8192','--max-rounds','8','--attempts','3',
 '--max-guqinizer-passes','3','--max-model-len','16384','--max-num-seqs','8',
 '--gpu-memory-utilization','0.75']
if r.get('rerun_stages','guqinizer') == 'guqinizer':
    command.extend(['--guqinizer-only-baseline',str(out/'baseline.jsonl')])
    for sid in r['target_ids']:
        command.extend(['--sample-id',sid])
else:
    assert r['rerun_stages'] == 'both'
    # Non-target records are already seeded as completed. Only missing
    # target IDs run through the ordinary two-stage inference path.
log=(out/'eval.log').open('a')
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
# Jobber's allocation is not a memory reservation: another process can start
# before vLLM loads. Require three consecutive healthy samples on our GPUs.
devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
if not all(devices):
    raise RuntimeError('CUDA_VISIBLE_DEVICES must identify allocated GPUs')
minimum=int(os.environ.get('REFINEMENT_MIN_FREE_MIB','20480'))
stable=0
while stable < 3:
    free=[]
    for device in devices:
        value=subprocess.check_output([
            'nvidia-smi','-i',device,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True)
        free.append(int(value.strip()))
    stable=stable+1 if all(v >= minimum for v in free) else 0
    print(json.dumps({'event':'gpu_memory_wait','devices':devices,
                      'free_mib':free,'required_mib':minimum,
                      'stable_samples':stable}),flush=True)
    if stable < 3:
        time.sleep(20)
os.execv(sys.executable,command)
PY
