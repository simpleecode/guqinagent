#!/usr/bin/env python3
"""Capture modern seed scores from the connected Sitongli Android app.

The Frida hook must attach before navigating to a score.  In particular, do
not use a helper that force-stops the app after the hook has attached: that
detaches the hook and gives a deceptively empty capture.

This tool intentionally saves raw runtime evidence even when the numbered
notation tonic cannot be proven from the captured global transpose value.  It
never guesses a tonic just to manufacture a mapped score.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# The eighth integer of the direct runtime Nab object is the App's global
# transpose/key-signature code.  It is orthogonal to the first seven per-string
# tuning offsets.  These values are verified against the App's displayed key
# labels; preserve all eight integers in the exported metadata.
TONIC_BY_GLOBAL_TRANSPOSE = {0: "F", 5: "C", 7: "B", -7: "C", -2: "G"}


def load_seeds(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"(?:scores/)([A-Za-z0-9]+)", line)
        if not match:
            raise ValueError(f"No score key in seed line: {line!r}")
        key = match.group(1)
        # The seed format is `<title> <share-url>`.  Do not retain the URL
        # as part of the UI title passed to the navigator.
        title = line[: match.start()].strip()
        title = title.rsplit(None, 1)[0].strip() if " " in title else title
        title = title or key
        yield title, key


def score_metadata(key: str) -> dict:
    url = f"https://api.sitongli.net/v9/scores/{key}/simple"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.load(response)
    # The v9 endpoint has shipped both an enveloped `{data: ...}` response
    # and a direct score object.  Accept both rather than treating public
    # scores as a capture failure.
    return payload.get("data", payload)


def wait_for_hook(process: subprocess.Popen, output: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Frida exited early ({process.returncode})")
        if output.exists() and '"name": "source_jab_event"' in output.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.2)
    raise TimeoutError("source_jab_event hook was not ready")


def runtime_transpose(capture: Path) -> int | None:
    pattern = re.compile(r'"off_c!GrowableList[^\n]*?\[([^\]]+)\]')
    for line in capture.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"name": "note_tuning' not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        try:
            values = [int(part.strip()) for part in match.group(1).split(",")]
        except ValueError:
            continue
        if len(values) == 8:
            return values[7]
    return None


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def capture_one(args, root: Path, title: str, key: str) -> str:
    score_dir = root / key
    evidence = score_dir / "evidence"
    raw = score_dir / "raw_data.json"
    if raw.exists():
        return f"skip-complete {key} {title}"
    evidence.mkdir(parents=True, exist_ok=True)
    meta = score_metadata(key)
    score_id = meta["id"]
    app_title = str(meta.get("title") or title)
    capture = evidence / "runtime_live.jsonl"
    hook_stdout = evidence / "frida_stdout.log"
    hook_stderr = evidence / "frida_stderr.log"
    hook_command = [
        # `wait_for_hook` reads the hook's redirected stdout.  Force Python
        # to flush its ready record immediately; otherwise a pipe can retain
        # it until process exit and cause a false timeout.
        args.python, "-u", "scripts/legacy_capture/frida_decode_score_jians.py",
        "--adb", args.adb, "--package", args.package,
        "--blutter-js", args.blutter_js, "--host", args.host,
        "--out", str(capture),
        "--only", "note_slur", "--only", "note_render",
        "--only", "note_tuning", "--only", "view_get_jians",
        "--only", "jianzi_component", "--offset", "source_jab_event=0x63f220",
        "--entry-only", "--depth", "10", "--array-limit", "4096", "--map-limit", "1024",
    ]
    # Cold-start the exact score detail *before* attaching.  The Gadget app
    # ignores a later VIEW intent when a Flutter instance is already topmost;
    # that previously caused a valid hook to capture the preceding score.
    run([args.adb, "shell", "am", "start", "-S", "-W", "-a",
         "android.intent.action.VIEW", "-d", f"sitongli://app/scores/{score_id}",
         args.package], cwd=args.cwd)
    time.sleep(args.delay)
    with hook_stdout.open("w", encoding="utf-8") as out, hook_stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(hook_command, cwd=args.cwd, stdout=out, stderr=err)
        try:
            wait_for_hook(proc, hook_stdout, args.hook_timeout)
            run([
                args.python, "scripts/legacy_capture/adb_navigate_score.py",
                "--adb", args.adb, "--package", args.package,
                "--deeplink", f"sitongli://app/scores/{score_id}",
                "--score-title", app_title, "--out-dir", str(evidence / "adb_nav"),
                "--keep-running", "--allow-missing-tuning-marker",
                "--skip-detail-title-check", "--already-on-detail",
                "--scrolls", str(args.scrolls), "--delay", str(args.delay),
            ], cwd=args.cwd)
            time.sleep(args.capture_settle)
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    transpose = runtime_transpose(capture)
    tonic = TONIC_BY_GLOBAL_TRANSPOSE.get(transpose)
    manifest = {
        "score_key": key, "score_id": score_id, "title": app_title,
        "runtime_global_transpose": transpose, "verified_tonic": tonic,
        "capture": str(capture),
    }
    (evidence / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if tonic is None:
        return f"captured-unmapped {key} {title}: transpose={transpose!r}"
    run([
        args.python, "scripts/extract_score_runtime_windows.py",
        "--input", str(capture), "--out-dir", str(score_dir),
        "--score-id", str(score_id), "--score-key", key, "--title", app_title,
        "--tonic", tonic, "--assume-complete-runtime-window",
    ], cwd=args.cwd)
    return f"captured-mapped {key} {title}: 1={tonic}, transpose={transpose}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="ABC_J/seeds/modern_seeds_eval.txt")
    parser.add_argument("--out-root", default="ABC_J/modern")
    parser.add_argument("--adb", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--package", default="com.sitongli.app.gadget")
    parser.add_argument("--blutter-js", default="work/blutter_out/blutter_frida.js")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--scrolls", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--capture-settle", type=float, default=2.0)
    # A cold attach on the connected phone routinely takes about 20 seconds.
    # Leave room for it rather than classifying a healthy hook as a failure.
    parser.add_argument("--hook-timeout", type=float, default=60.0)
    parser.add_argument("--keys", nargs="*", default=None,
                        help="Optional score keys to capture; useful for retrying failures only.")
    args = parser.parse_args()
    args.cwd = Path.cwd()
    selected = set(args.keys or [])
    for title, key in load_seeds(Path(args.seeds)):
        if selected and key not in selected:
            continue
        try:
            print(capture_one(args, Path(args.out_root), title, key), flush=True)
        except Exception as exc:
            print(f"failed {key} {title}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
