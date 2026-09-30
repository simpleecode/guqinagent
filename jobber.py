#!/usr/bin/env python3
"""Small persistent GPU job queue for the CIP lab server.

This scheduler is deliberately conservative: it only starts a queued job when
the requested number of GPUs are empty according to ``nvidia-smi``.  It owns
only the jobs submitted through this file; existing manual jobs are never
killed or otherwise managed.

Run ``python training/jobber.py --help`` for examples.  Jobs are
executed in their own process session, with CUDA_VISIBLE_DEVICES assigned by
the scheduler.  Submitted commands must stay in the foreground (do not append
``&``, ``nohup`` or ``setsid`` yourself).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

try:  # The scheduler runs on Linux; this fallback keeps --help usable locally.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows-only convenience
    fcntl = None


DEFAULT_ROOT = Path(__file__).resolve().parent / "runtime" / "gpu_queue"
FREE_MEMORY_MIB = 512


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "state": root / "state.json",
        "lock": root / "state.lock",
        "daemon_lock": root / "daemon.lock",
        "pid": root / "daemon.pid",
        "daemon_log": root / "daemon.log",
        "logs": root / "logs",
        "exits": root / "exits",
    }


def ensure_dirs(root: Path) -> None:
    for value in paths(root).values():
        if value.suffix:
            continue
        value.mkdir(parents=True, exist_ok=True)


def blank_state() -> dict[str, Any]:
    return {"version": 1, "paused": False, "next_id": 1, "jobs": []}


@contextlib.contextmanager
def state_guard(root: Path) -> Iterator[dict[str, Any]]:
    ensure_dirs(root)
    p = paths(root)
    with p["lock"].open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if p["state"].exists():
                state = json.loads(p["state"].read_text(encoding="utf-8"))
            else:
                state = blank_state()
            yield state
            tmp = p["state"].with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, p["state"])
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_job(state: dict[str, Any], job_id: str) -> dict[str, Any]:
    for job in state["jobs"]:
        if job["id"] == job_id:
            return job
    raise SystemExit(f"unknown job: {job_id}")


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def nvidia_gpus() -> list[dict[str, int]]:
    """Return GPU index/memory. Raises on unavailable or malformed nvidia-smi."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    result: list[dict[str, int]] = []
    for line in output.splitlines():
        index, memory = [part.strip() for part in line.split(",", maxsplit=1)]
        result.append({"index": int(index), "memory_mib": int(memory)})
    if not result:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return result


def exit_path(root: Path, job_id: str) -> Path:
    return paths(root)["exits"] / f"{job_id}.exit"


def log_path(root: Path, job_id: str) -> Path:
    return paths(root)["logs"] / f"{job_id}.log"


def refresh_jobs(state: dict[str, Any], root: Path) -> None:
    """Collect completed child sessions using their durable exit-code files."""
    for job in state["jobs"]:
        if job["status"] not in {"running", "cancelling"}:
            continue
        marker = exit_path(root, job["id"])
        if marker.exists():
            try:
                return_code = int(marker.read_text(encoding="utf-8").strip())
            except ValueError:
                return_code = 1
            if job["status"] == "cancelling":
                job["status"] = "cancelled"
            else:
                job["status"] = "succeeded" if return_code == 0 else "failed"
            job["return_code"] = return_code
            job["finished_at"] = utc_now()
        elif not pid_alive(job.get("pid")):
            job["status"] = "failed" if job["status"] == "running" else "cancelled"
            job["return_code"] = None
            job["finished_at"] = utc_now()
            job["error"] = "process disappeared without an exit marker"


def free_gpu_ids(state: dict[str, Any], memory_limit: int) -> list[int]:
    managed = {
        gpu
        for job in state["jobs"]
        if job["status"] in {"running", "cancelling"}
        for gpu in job.get("assigned_gpus", [])
    }
    return [gpu["index"] for gpu in nvidia_gpus() if gpu["memory_mib"] <= memory_limit and gpu["index"] not in managed]


def build_wrapper(job: dict[str, Any], root: Path) -> str:
    env_parts = [f"export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str, job['assigned_gpus'])))}"]
    for key, value in job.get("env", {}).items():
        env_parts.append(f"export {key}={shlex.quote(value)}")
    marker = exit_path(root, job["id"])
    # The marker is written only after the foreground command exits.  It lets
    # a restarted daemon recover the terminal state without owning Popen().
    return "\n".join([
        "set +e",
        f"cd {shlex.quote(job['cwd'])}",
        *env_parts,
        job["command"],
        "queue_rc=$?",
        f"printf '%s\\n' \"$queue_rc\" > {shlex.quote(str(marker) + '.tmp')}",
        f"mv {shlex.quote(str(marker) + '.tmp')} {shlex.quote(str(marker))}",
        "exit $queue_rc",
    ])


def launch_job(state: dict[str, Any], job: dict[str, Any], root: Path, assigned: list[int]) -> None:
    job["assigned_gpus"] = assigned
    marker = exit_path(root, job["id"])
    marker.unlink(missing_ok=True)
    log = log_path(root, job["id"])
    wrapper = build_wrapper(job, root)
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            ["bash", "-lc", wrapper],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=job["cwd"],
        )
    job.update({
        "status": "running",
        "pid": process.pid,
        "pgid": process.pid,
        "started_at": utc_now(),
        "finished_at": None,
        "return_code": None,
        "error": None,
        "log_path": str(log),
    })


def schedule_once(root: Path, memory_limit: int) -> None:
    with state_guard(root) as state:
        refresh_jobs(state, root)
        if state.get("paused"):
            return
        queued = [job for job in state["jobs"] if job["status"] == "queued"]
        if not queued:
            return

        jobs_by_id = {job["id"]: job for job in state["jobs"]}
        runnable: list[dict[str, Any]] = []
        for job in queued:
            dependencies = job.get("depends_on", [])
            missing = [job_id for job_id in dependencies if job_id not in jobs_by_id]
            if missing:
                job["last_wait_reason"] = f"blocked by missing dependencies: {', '.join(missing)}"
                continue
            failed = [
                job_id for job_id in dependencies
                if jobs_by_id[job_id]["status"] in {"failed", "cancelled"}
            ]
            if failed:
                job["last_wait_reason"] = f"blocked by failed dependencies: {', '.join(failed)}"
                continue
            pending = [
                job_id for job_id in dependencies
                if jobs_by_id[job_id]["status"] != "succeeded"
            ]
            if pending:
                job["last_wait_reason"] = f"waiting for dependencies: {', '.join(pending)}"
                continue
            runnable.append(job)

        if not runnable:
            return

        # Priority is a scheduling barrier: a lower-priority job may not
        # backfill around a runnable higher-priority job.  Within the highest
        # priority level, scan in queue order so same-priority jobs can still
        # backfill when an earlier job needs more GPUs than are available.
        highest_priority = max(int(job.get("priority", 0)) for job in runnable)
        candidates = [
            job for job in runnable
            if int(job.get("priority", 0)) == highest_priority
        ]
        try:
            available = free_gpu_ids(state, memory_limit)
        except Exception as exc:
            reason = f"GPU probe failed: {type(exc).__name__}: {exc}"
            for job in candidates:
                job["last_wait_reason"] = reason
            return

        for job in candidates:
            if len(available) < job["gpus"]:
                job["last_wait_reason"] = (
                    f"waiting for {job['gpus']} GPUs; {len(available)} available under "
                    f"{memory_limit} MiB"
                )
                continue
            assigned = available[: job["gpus"]]
            launch_job(state, job, root, assigned)
            available = available[job["gpus"] :]


def daemon_is_running(root: Path) -> bool:
    pid_file = paths(root)["pid"]
    if not pid_file.exists():
        return False
    try:
        return pid_alive(int(pid_file.read_text(encoding="utf-8").strip()))
    except ValueError:
        return False


def serve(root: Path, poll_seconds: float, memory_limit: int) -> None:
    ensure_dirs(root)
    p = paths(root)
    with p["daemon_lock"].open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("gpu queue daemon is already running")
        p["pid"].write_text(str(os.getpid()) + "\n", encoding="utf-8")
        stop = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            while not stop:
                try:
                    schedule_once(root, memory_limit)
                except Exception as exc:  # keep the service alive; record a useful trace
                    with p["daemon_log"].open("a", encoding="utf-8") as handle:
                        handle.write(f"{utc_now()} scheduler error: {type(exc).__name__}: {exc}\n")
                time.sleep(poll_seconds)
        finally:
            if p["pid"].exists() and p["pid"].read_text(encoding="utf-8").strip() == str(os.getpid()):
                p["pid"].unlink()


def start_daemon(root: Path, poll_seconds: float, memory_limit: int) -> None:
    ensure_dirs(root)
    if daemon_is_running(root):
        print(f"daemon already running (pid {paths(root)['pid'].read_text().strip()})")
        return
    with paths(root)["daemon_log"].open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "serve", "--root", str(root),
             "--poll-seconds", str(poll_seconds), "--free-memory-mib", str(memory_limit)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(0.2)
    if not pid_alive(process.pid):
        raise SystemExit(f"daemon failed to start; inspect {paths(root)['daemon_log']}")
    print(f"started daemon pid {process.pid}")


def parse_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key.replace("_", "a").isalnum() or key[:1].isdigit():
            raise SystemExit(f"invalid environment key: {key}")
        result[key] = value
    return result


def command_text(parts: list[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        raise SystemExit("submit requires a foreground command after --")
    return parts[0] if len(parts) == 1 else shlex.join(parts)


def submit(args: argparse.Namespace) -> None:
    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")
    root = args.root.resolve()
    start_daemon(root, args.poll_seconds, args.free_memory_mib)
    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.is_dir():
        raise SystemExit(f"--cwd does not exist or is not a directory: {cwd}")
    with state_guard(root) as state:
        depends_on = list(dict.fromkeys(args.depends_on))
        known_job_ids = {job["id"] for job in state["jobs"]}
        unknown = [job_id for job_id in depends_on if job_id not in known_job_ids]
        if unknown:
            raise SystemExit(f"unknown dependency job id(s): {', '.join(unknown)}")
        job_id = f"J{state['next_id']:05d}"
        state["next_id"] += 1
        job = {
            "id": job_id,
            "name": args.name or job_id,
            "gpus": args.gpus,
            "priority": args.priority,
            "depends_on": depends_on,
            "command": command_text(args.command),
            "cwd": str(cwd),
            "env": parse_env(args.env),
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "assigned_gpus": [],
            "pid": None,
            "pgid": None,
            "return_code": None,
            "error": None,
            "last_wait_reason": "queued",
            "log_path": str(log_path(root, job_id)),
        }
        if args.front:
            state["jobs"].insert(0, job)
        elif args.before:
            target = get_job(state, args.before)
            if target["status"] != "queued":
                raise SystemExit("--before target must still be queued")
            state["jobs"].insert(state["jobs"].index(target), job)
        else:
            state["jobs"].append(job)
    print(f"submitted {job_id}: {job['name']} ({job['gpus']} GPU(s))")


def show_status(args: argparse.Namespace) -> None:
    with state_guard(args.root.resolve()) as state:
        refresh_jobs(state, args.root.resolve())
        print(f"daemon={'running' if daemon_is_running(args.root.resolve()) else 'stopped'}  paused={state.get('paused', False)}")
        print("ID      STATUS      GPUS  ASSIGNED  PRIO  NAME")
        for job in state["jobs"]:
            assigned = ",".join(map(str, job.get("assigned_gpus", []))) or "-"
            print(f"{job['id']:<7} {job['status']:<11} {job['gpus']:<5} {assigned:<9} {job.get('priority', 0):<5} {job['name']}")
            if args.verbose:
                print(f"  pid={job.get('pid')} pgid={job.get('pgid')} started={job.get('started_at')}\n  cwd={job['cwd']}\n  log={job['log_path']}\n  command={job['command']}")
                if job.get("depends_on"):
                    print(f"  depends_on={','.join(job['depends_on'])}")
                if job.get("last_wait_reason"):
                    print(f"  note={job['last_wait_reason']}")


def show_processes(args: argparse.Namespace) -> None:
    """Show the complete process session for one job or all running jobs."""
    root = args.root.resolve()
    with state_guard(root) as state:
        refresh_jobs(state, root)
        jobs = [job for job in state["jobs"] if job["status"] in {"running", "cancelling"}]
        if args.job_id:
            jobs = [get_job(state, args.job_id)]
    if not jobs:
        print("no running jobs")
        return
    for job in jobs:
        print(f"--- {job['id']} {job['name']} status={job['status']} pid={job.get('pid')} gpus={','.join(map(str, job.get('assigned_gpus', [])))}")
        pid = job.get("pid")
        if not pid:
            print("(no pid recorded)")
            continue
        try:
            output = subprocess.check_output(
                ["ps", "-o", "pid=,ppid=,pgid=,stat=,etime=,%cpu=,%mem=,cmd=", "--sid", str(pid)],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            output = f"process query failed: {exc}"
        print(output or "process session no longer exists")


def cancel(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    with state_guard(root) as state:
        refresh_jobs(state, root)
        job = get_job(state, args.job_id)
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = utc_now()
            print(f"cancelled queued job {job['id']}")
            return
        if job["status"] not in {"running", "cancelling"}:
            print(f"job {job['id']} is already {job['status']}")
            return
        sig = signal.SIGKILL if args.force else signal.SIGTERM
        try:
            os.killpg(int(job.get("pgid") or job["pid"]), sig)
        except ProcessLookupError:
            pass
        job["status"] = "cancelling"
        job["cancel_requested_at"] = utc_now()
        print(f"sent {sig.name} to {job['id']} (use cancel --force only if TERM does not stop it)")


def move(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    with state_guard(root) as state:
        job = get_job(state, args.job_id)
        if job["status"] != "queued":
            raise SystemExit("only queued jobs can be reordered")
        state["jobs"].remove(job)
        if args.front:
            state["jobs"].insert(0, job)
        elif args.back:
            state["jobs"].append(job)
        else:
            target = get_job(state, args.before)
            if target["status"] != "queued":
                raise SystemExit("--before target must be queued")
            state["jobs"].insert(state["jobs"].index(target), job)
    print(f"moved {args.job_id}")


def resize(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")
    with state_guard(root) as state:
        job = get_job(state, args.job_id)
        if job["status"] != "queued":
            raise SystemExit("only queued jobs can change GPU requirements")
        old = job["gpus"]
        job["gpus"] = args.gpus
        job["last_wait_reason"] = f"GPU requirement changed from {old} to {args.gpus}"
    print(f"resized {args.job_id}: {old} -> {args.gpus} GPU(s)")


def pause(args: argparse.Namespace, paused: bool) -> None:
    with state_guard(args.root.resolve()) as state:
        state["paused"] = paused
    print("queue paused" if paused else "queue resumed")


def stop_daemon(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    p = paths(root)["pid"]
    if not p.exists():
        print("daemon is not running")
        return
    pid = int(p.read_text(encoding="utf-8").strip())
    if pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        print(f"sent TERM to daemon {pid}; running jobs are left untouched")
    else:
        p.unlink(missing_ok=True)
        print("removed stale daemon pid file")


def show_logs(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    with state_guard(root) as state:
        job = get_job(state, args.job_id)
        path = Path(job["log_path"])
    if not path.exists():
        raise SystemExit(f"log not yet created: {path}")
    print(path)
    if args.tail:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-args.tail:]))


def doctor(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    print(f"queue_root={root}")
    print(f"daemon_running={daemon_is_running(root)}")
    try:
        for gpu in nvidia_gpus():
            print(f"gpu={gpu['index']} memory_used_mib={gpu['memory_mib']} free={gpu['memory_mib'] <= args.free_memory_mib}")
    except Exception as exc:
        print(f"gpu_probe_error={type(exc).__name__}: {exc}")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="persistent queue directory")
    common.add_argument("--free-memory-mib", type=int, default=FREE_MEMORY_MIB,
                        help="GPU is considered free only at or below this used-memory threshold")
    # Short polling reduces the race window in which an external process can
    # acquire a just-freed GPU. The memory threshold remains the safety guard.
    common.add_argument("--poll-seconds", type=float, default=2.0)
    p = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("serve", parents=[common], help="run the queue daemon in the foreground")
    sub.add_parser("start", parents=[common], help="start the persistent queue daemon")
    sub.add_parser("stop", parents=[common], help="stop daemon; do not terminate jobs")
    sub.add_parser("status", parents=[common], help="show queue state").add_argument("--verbose", action="store_true")
    sub.add_parser("doctor", parents=[common], help="show GPU availability")
    pp = sub.add_parser("ps", parents=[common], help="show process tree for running jobs")
    pp.add_argument("job_id", nargs="?", help="optional job id; omit to show all running jobs")
    sub.add_parser("pause", parents=[common], help="prevent new jobs from starting")
    sub.add_parser("resume", parents=[common], help="allow queued jobs to start")

    sp = sub.add_parser("submit", parents=[common], help="enqueue a foreground command")
    sp.add_argument("--gpus", type=int, required=True)
    sp.add_argument("--name", default="")
    sp.add_argument("--priority", type=int, default=0,
                    help="higher values run first and block lower-priority backfill")
    sp.add_argument("--depends-on", action="append", default=[], metavar="JOB_ID",
                    help="require a job to succeed before this job can start; repeatable")
    sp.add_argument("--cwd", default=str(Path.cwd()))
    sp.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    group = sp.add_mutually_exclusive_group()
    group.add_argument("--front", action="store_true", help="insert before all queued work")
    group.add_argument("--before", metavar="JOB_ID", help="insert before a queued job")
    sp.add_argument("command", nargs=argparse.REMAINDER, help="command after --")

    cp = sub.add_parser("cancel", parents=[common], help="cancel a queued/running job")
    cp.add_argument("job_id")
    cp.add_argument("--force", action="store_true", help="send SIGKILL instead of SIGTERM")
    mp = sub.add_parser("move", parents=[common], help="reorder a queued job")
    mp.add_argument("job_id")
    group = mp.add_mutually_exclusive_group(required=True)
    group.add_argument("--front", action="store_true")
    group.add_argument("--back", action="store_true")
    group.add_argument("--before", metavar="JOB_ID")
    rp = sub.add_parser("resize", parents=[common], help="change GPU count for a queued job")
    rp.add_argument("job_id")
    rp.add_argument("--gpus", type=int, required=True)
    lp = sub.add_parser("logs", parents=[common], help="show log location or final lines")
    lp.add_argument("job_id")
    lp.add_argument("--tail", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == "serve":
        serve(args.root.resolve(), args.poll_seconds, args.free_memory_mib)
    elif args.action == "start":
        start_daemon(args.root.resolve(), args.poll_seconds, args.free_memory_mib)
    elif args.action == "stop":
        stop_daemon(args)
    elif args.action == "submit":
        submit(args)
    elif args.action == "status":
        show_status(args)
    elif args.action == "cancel":
        cancel(args)
    elif args.action == "move":
        move(args)
    elif args.action == "resize":
        resize(args)
    elif args.action == "pause":
        pause(args, True)
    elif args.action == "resume":
        pause(args, False)
    elif args.action == "logs":
        show_logs(args)
    elif args.action == "doctor":
        doctor(args)
    elif args.action == "ps":
        show_processes(args)


if __name__ == "__main__":
    main()
