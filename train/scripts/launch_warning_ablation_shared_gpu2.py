"""Launch the authorized OFF continuation directly on shared GPU2."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'train/eval_outputs/warning_ablation/full/off'


def main():
    record = OUT / 'direct_gpu2_launch.json'
    assert not record.exists(), 'Already launched; inspect recorded PID instead of duplicating'
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 8992)) != 0, 'Existing inference server on port 8992'
    free = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 20480, f'Not enough free memory: {free} MiB'
    baseline = {}
    for path in sorted((OUT / 'workers').glob('predictions_*.jsonl')):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row['sample_id']
            assert sid not in baseline and row['protocol_valid']
            baseline[sid] = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert len(baseline) == 940
    archive = OUT / 'logs_before_direct_gpu2'
    archive.mkdir()
    for path in list((OUT / 'workers').glob('worker_*.log')) + [OUT / 'vllm_server.log']:
        if path.exists():
            shutil.copy2(path, archive / path.name)
    subprocess.run([sys.executable, str(ROOT / 'train/scripts/prepare_warning_ablation.py'), '--scope', 'full'], check=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'
    log_path = OUT / 'direct_gpu2.log'
    with log_path.open('ab') as log:
        proc = subprocess.Popen(['bash', str(ROOT / 'train/scripts/run_warning_ablation_full.sh'), 'off'],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    report = {'pid': proc.pid, 'pgid': proc.pid, 'gpu': 2, 'free_mib_before_launch': free,
              'baseline_count': len(baseline), 'remaining': 39, 'baseline_row_sha256': baseline,
              'superseded_job': 'J00355', 'log': str(log_path), 'other_processes_untouched': True}
    record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'baseline_row_sha256'}), flush=True)


if __name__ == '__main__':
    main()
