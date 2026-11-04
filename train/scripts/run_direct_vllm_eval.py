#!/usr/bin/env python3
"""Launch one local vLLM replica for a trained Direct SFT checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
import socket
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.scripts.run_vllm_eval_queue import stop_server_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--vllm-env', default='guqin-vllm-019')
    parser.add_argument('--port', type=int, default=8999)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    adapter_sha = hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir/'.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output_dir/'predictions.jsonl').exists() and not args.resume:
        raise ValueError('Existing results require explicit --resume')
    probe = socket.socket()
    try:
        # The previous scope's server can leave TIME_WAIT sockets after a
        # clean shutdown. Reuse those, while still rejecting active listeners.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    finally:
        probe.close()
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    stable = 0
    while stable < 3:
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        stable = stable + 1 if free >= 20480 else 0
        print(f'gpu_memory_wait free_mib={free} stable_samples={stable}', flush=True)
        if stable < 3:
            time.sleep(20)
    conda = str(Path.home()/'miniconda3/bin/conda')
    server_command = [conda, 'run', '--no-capture-output', '-n', args.vllm_env,
        'vllm', 'serve', args.base_model, '--served-model-name', 'qwen35-base',
        '--port', str(args.port), '--dtype', 'bfloat16', '--quantization', 'bitsandbytes',
        '--max-model-len', '16384', '--max-num-seqs', '8', '--gpu-memory-utilization', '0.75',
        '--enforce-eager', '--enable-lora', '--lora-modules', f'guqin-direct={args.adapter}',
        '--max-lora-rank', '8', '--max-loras', '1']
    log = (args.output_dir/'vllm_server.log').open('a', encoding='utf-8')
    server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    def terminate(_signum, _frame):
        raise SystemExit('Direct evaluator terminated')
    signal.signal(signal.SIGTERM, terminate)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError('vLLM exited; inspect vllm_server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=3):
                    break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('vLLM did not become healthy')
                time.sleep(3)
        command = [sys.executable, str(ROOT/'train/scripts/eval_direct.py'),
            '--input', str(args.input), '--output-dir', str(args.output_dir),
            '--vllm-url', f'http://127.0.0.1:{args.port}/v1', '--workers', str(args.workers),
            '--checkpoint-sha256', adapter_sha]
        if args.resume:
            command.append('--resume')
        subprocess.run(command, check=True, cwd=ROOT)
    finally:
        stop_server_group(server, log)
        lock.close()


if __name__ == '__main__':
    main()
