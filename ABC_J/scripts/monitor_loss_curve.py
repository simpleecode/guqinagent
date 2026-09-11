#!/usr/bin/env python3
"""Headless live loss monitor for a local file or a remote log over SSH.

Examples:
  python ABC_J/scripts/monitor_loss_curve.py \
    --remote-log /home/20223393ljw/code/training/runtime/gpu_queue/logs/J00115.log

  python ABC_J/scripts/monitor_loss_curve.py --local-log /path/to/train.log
"""

from __future__ import annotations

import argparse
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path


LOSS_RE = re.compile(
    r"(?:['\"]?loss['\"]?)\s*[:=]\s*"
    r"['\"]?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)['\"]?",
    re.IGNORECASE,
)
STEP_PATTERNS = (
    re.compile(r"(?:global_step|global step)\s*[:=]\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bstep\s*[:=]\s*(\d+)", re.IGNORECASE),
)
PROGRESS_STEP_RE = re.compile(r"\|\s*(\d+)\s*/\s*\d+\s*\[")


def parse_loss(line: str, fallback_step: int) -> tuple[int, float] | None:
    """Extract one loss point from a Trainer/logging line."""
    match = LOSS_RE.search(line)
    if not match:
        return None
    try:
        loss = float(match.group(1))
    except ValueError:
        return None

    step = None
    for pattern in STEP_PATTERNS:
        step_match = pattern.search(line)
        if step_match:
            step = int(step_match.group(1))
            break
    return (fallback_step if step is None else step, loss)


def read_history(lines: list[str]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    progress_step = 0
    for index, line in enumerate(lines, start=1):
        progress_match = PROGRESS_STEP_RE.search(line)
        if progress_match:
            progress_step = int(progress_match.group(1))
        point = parse_loss(line, progress_step or index)
        if point is not None:
            points.append(point)
    return points


def follow_local(path: Path, history: int, interval: float, output: queue.Queue[str | None], stop: threading.Event) -> None:
    """Read existing lines and then follow a local log file."""
    while not path.exists() and not stop.wait(interval):
        pass
    if stop.is_set():
        output.put(None)
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        initial = deque(handle, maxlen=history)
        for line in initial:
            output.put(line)
        while not stop.is_set():
            line = handle.readline()
            if line:
                output.put(line)
            else:
                stop.wait(interval)
    output.put(None)


def ssh_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    destination = f"{args.ssh_user}@{args.ssh_host}"
    return [
        "ssh",
        "-p",
        str(args.ssh_port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        destination,
        remote_command,
    ]


def follow_remote(args: argparse.Namespace, output: queue.Queue[str | None], stop: threading.Event) -> subprocess.Popen[str]:
    remote_command = f"tail -n {args.history} -F -- {shlex.quote(args.remote_log)}"
    process = subprocess.Popen(
        ssh_command(args, remote_command),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

    def read_stream() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if stop.is_set():
                break
            output.put(line)
        output.put(None)

    threading.Thread(target=read_stream, daemon=True).start()
    return process


def fetch_remote_once(args: argparse.Namespace) -> list[str]:
    remote_command = f"tail -n {args.history} -- {shlex.quote(args.remote_log)}"
    process = subprocess.Popen(
        ssh_command(args, remote_command),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    stdout, _ = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"ssh/tail exited with status {process.returncode}")
    return stdout.splitlines()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-log", type=Path, help="local training log to follow")
    source.add_argument("--remote-log", help="remote training log to follow over SSH")
    parser.add_argument("--ssh-host", default="219.216.65.119")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-user", default="20223393ljw")
    parser.add_argument("--history", type=int, default=300, help="number of existing log lines to load")
    parser.add_argument("--interval", type=float, default=2.0, help="local-file polling interval in seconds")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path.home() / "Desktop" / "guqin_sft_loss.png",
        help="PNG path; the same file is overwritten after each new loss (default: ~/Desktop/guqin_sft_loss.png)",
    )
    parser.add_argument("--once", action="store_true", help="parse once and print the latest loss without opening a plot")
    return parser


def build_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("需要 matplotlib：python3 -m pip install matplotlib") from exc
    figure, axis = plt.subplots(figsize=(10, 5.5))
    line, = axis.plot([], [], linewidth=1.8, label="loss")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("Training loss (waiting for log output)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return plt, figure, axis, line


def update_plot(axis, line, points: list[tuple[int, float]], figure) -> None:
    if not points:
        return
    steps, losses = zip(*points)
    line.set_data(steps, losses)
    axis.relim()
    axis.autoscale_view()
    axis.set_title(f"Training loss — step {steps[-1]} — loss {losses[-1]:.6f}")
    figure.canvas.draw_idle()


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def run_once(args: argparse.Namespace) -> int:
    if args.local_log:
        if not args.local_log.exists():
            raise SystemExit(f"日志不存在：{args.local_log}")
        lines = args.local_log.read_text(encoding="utf-8", errors="replace").splitlines()[-args.history :]
    else:
        lines = fetch_remote_once(args)
    points = read_history(lines)
    if not points:
        print("未解析到 loss；日志可能还没有输出训练步。")
        return 0
    step, loss = points[-1]
    print(f"points={len(points)} latest_step={step} latest_loss={loss:.8f}")
    return 0


def run_live(args: argparse.Namespace) -> int:
    source = str(args.local_log) if args.local_log else f"{args.ssh_user}@{args.ssh_host}:{args.remote_log}"
    print(f"开始监控：{source}", flush=True)
    print(f"曲线文件：{args.save}（持续覆盖）", flush=True)
    try:
        plt, figure, axis, line = build_plot()
    except SystemExit:
        raise

    output: queue.Queue[str | None] = queue.Queue()
    stop = threading.Event()
    process: subprocess.Popen[str] | None = None
    if args.local_log:
        threading.Thread(
            target=follow_local,
            args=(args.local_log, args.history, args.interval, output, stop),
            daemon=True,
        ).start()
    else:
        process = follow_remote(args, output, stop)
        print("SSH tail 已启动；若暂时没有新 loss，脚本会保持等待。", flush=True)

    points: list[tuple[int, float]] = []
    fallback_step = 0
    progress_step = 0
    try:
        while not stop.is_set():
            got_line = False
            stream_done = False
            while True:
                try:
                    item = output.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stop.set()
                    stream_done = True
                    break
                progress_match = PROGRESS_STEP_RE.search(item)
                if progress_match:
                    progress_step = int(progress_match.group(1))
                fallback_step += 1
                point = parse_loss(item, progress_step or fallback_step)
                if point is not None:
                    points.append(point)
                    step, loss = point
                    print(f"step={step} loss={loss:.8f}", flush=True)
                    got_line = True
            if got_line:
                update_plot(axis, line, points, figure)
                if args.save:
                    save_figure(figure, args.save)
            if stream_done or (process is not None and process.poll() is not None and output.empty()):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n已停止 loss 监控。")
    finally:
        stop.set()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        if args.save and points:
            save_figure(figure, args.save)
        plt.close(figure)
    return 0


def main() -> int:
    args = make_parser().parse_args()
    if args.history < 1:
        raise SystemExit("--history 必须大于 0")
    if args.once:
        return run_once(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
