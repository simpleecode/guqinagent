"""Measure the validated redacted candidate using the full remote Qwen template."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'ABC_J/agent_training/single_stage_audited/.quality_rebuild'
HOST = '20223393ljw@219.216.65.119'
PYTHON = '/home/20223393ljw/miniconda3/envs/guqin-vllm-019/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = json.loads((BUILD / 'candidate_validation_report.json').read_text())
    assert validation['status'] == 'candidate_validated_not_promoted'
    candidate = BUILD / 'redaction_work/sft_candidate'
    data = candidate / 'guqin_agent_train.jsonl'
    metadata = candidate / 'guqin_agent_train_metadata.jsonl'
    assert sha(data) == validation['sft_sha256']
    temp = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', HOST,
                                   'mktemp -d /tmp/guqin-single-quality.XXXXXX'], text=True).strip()
    assert re.fullmatch(r'/tmp/guqin-single-quality\.[A-Za-z0-9]+', temp)
    script = ROOT / 'train/scripts/measure_token_distribution.py'
    # Only public, redacted SFT is uploaded. Private teacher audit stays local.
    subprocess.run(['rsync', '-a', str(data), str(metadata), str(script), f'{HOST}:{temp}/'], check=True)
    command = [PYTHON, temp + '/measure_token_distribution.py', '--input', temp + '/' + data.name,
               '--metadata', temp + '/' + metadata.name, '--model-path', '/home/20223393ljw/models/Qwen3.5-9B',
               '--output', temp + '/candidate_token_distribution.json',
               '--over-threshold-output', temp + '/candidate_over_8192.tsv', '--threshold', '8192']
    subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, shlex.join(command)], check=True)
    subprocess.run(['rsync', '-a', f'{HOST}:{temp}/candidate_token_distribution.json',
                    f'{HOST}:{temp}/candidate_over_8192.tsv', str(BUILD) + '/'], check=True)
    assert sha(data) == validation['sft_sha256']
    report = {'sft_sha256': sha(data), 'metadata_sha256': sha(metadata), 'measurement_script_sha256': sha(script),
              'processor_model': '/home/20223393ljw/models/Qwen3.5-9B', 'threshold': 8192,
              'remote_temporary_directory': temp, 'private_audit_uploaded': False,
              'report_sha256': sha(BUILD / 'candidate_token_distribution.json')}
    (BUILD / 'candidate_length_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Fresh candidate length measurement verified; formal data still unchanged.', flush=True)


if __name__ == '__main__':
    main()
