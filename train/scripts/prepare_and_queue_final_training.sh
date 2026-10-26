#!/usr/bin/env bash
set -euo pipefail
cd /home/20223393ljw/guqin-agent
source /home/20223393ljw/miniconda3/etc/profile.d/conda.sh
conda activate guqin-sft
DATA=train/data/guqin_agent_sft_two_stage_final
printf '%s  %s\n' 6ea8329df61ade844389611d76ba5e3215deba19f9a9aa6e7e0ae1d8c8d150b6 "$DATA/guqin_agent_train.jsonl" | sha256sum --check --status
export CUDA_VISIBLE_DEVICES=""
python train/scripts/validate_sft_export.py --input-dir "$DATA"
python train/scripts/verify_test50_training_inputs.py --dataset-dir "$DATA" --expected-rows 7952
python train/scripts/preflight_sft.py --input "$DATA/guqin_agent_train.jsonl" \
  --model-path /home/20223393ljw/models/Qwen3.5-9B --cutoff-len 8192 \
  --report "$DATA/preflight_report.json"
python train/scripts/prepare_training_tokenized_cache.py \
  --config train/configs/qwen35_9b_lora_two_stage_final_3gpu_bs12.yaml
python - <<'PY'
import json,subprocess
from pathlib import Path
jobber=['/home/20223393ljw/miniconda3/envs/cip/bin/python','/home/20223393ljw/code/training/jobber.py']
def submit(args):
    result=subprocess.check_output(jobber+args,text=True)
    print(result,flush=True)
    import re
    match=re.search(r'submitted (J\d+):',result)
    if not match:raise RuntimeError(result)
    return match[1]
train=submit(['submit','--gpus','3','--free-memory-mib','4096','--name','guqin-two-stage-final-e3-bs12-3gpu',
              '--cwd','/home/20223393ljw/guqin-agent','--','/bin/bash','train/scripts/run_two_stage_final_train_3gpu.sh'])
evaluation=submit(['submit','--gpus','4','--free-memory-mib','4096','--depends-on',train,
                   '--name','guqin-test50-two-stage-final-full-4gpu','--cwd','/home/20223393ljw/guqin-agent',
                   '--','/bin/bash','train/scripts/run_test50_two_stage_final_full_4gpu.sh'])
subprocess.run(jobber+['cancel','J00310'],check=True)
Path('train/data/guqin_agent_sft_two_stage_final/queued_jobs.json').write_text(json.dumps(
    {'training_job':train,'evaluation_job':evaluation,'superseded_evaluation_job':'J00310',
     'dataset_sha256':'6ea8329df61ade844389611d76ba5e3215deba19f9a9aa6e7e0ae1d8c8d150b6'},indent=2)+'\n')
PY
