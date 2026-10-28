"""Start the authorized sequential Youlan ON/OFF run after full OFF completes."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT / 'train/eval_outputs/warning_ablation/youlan'
    record = out / 'launch.json'
    assert not record.exists(), 'Already launched; do not duplicate'
    summary = json.loads((out.parent / 'full/off/run_summary.json').read_text())
    assert summary['completed'] == summary['protocol_valid'] == summary['expected'] == 979
    assert summary['duplicate_records_discarded'] == summary['skipped_after_invalid_predecessor'] == 0
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 8993)) != 0
    subprocess.run([sys.executable, str(ROOT / 'train/scripts/prepare_warning_ablation.py'), '--scope', 'youlan'], check=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'
    with (out / 'coordinator.log').open('ab') as log:
        process = subprocess.Popen(['bash', 'train/scripts/run_youlan_warning_ablation.sh'], cwd=ROOT,
                                   env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    report = {'pid': process.pid, 'pgid': process.pid, 'gpu': 2, 'arms': ['on', 'off'],
              'phrases_per_arm': 52, 'mode': 'sequential, same score history, one client',
              'prerequisite_full_off': summary, 'private_references_uploaded': False}
    record.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
