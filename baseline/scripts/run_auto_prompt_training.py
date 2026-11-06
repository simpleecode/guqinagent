#!/usr/bin/env python3
"""One command: load .env, check the pilot, evolve prompts, then confirm safely."""
from __future__ import annotations

import argparse
import ctypes
import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "guqinagent"
sys.path[:0] = [str(WORKSPACE / ".gepa_packages"), str(REPO), str(REPO / "baseline/gepa/src")]
SCRIPTS = REPO / "baseline/scripts"
RUNS = REPO / "baseline/runs"
ENV_FILE = REPO / ".env"
PLAN = REPO / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/gepa_pilot_manifest.json"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    key = os.environ.get("GLM_API_KEY", "")
    return text.replace(key, "[API_KEY_REDACTED]") if key else text


def run_path(name: str) -> Path:
    if name in {".", ".."} or not re.fullmatch(r"[\w.-]+", name):
        raise ValueError("RunName只能包含中文、字母、数字、下划线、连字符和点，不能包含路径分隔符。")
    result = (RUNS / name).resolve()
    if result.parent != RUNS.resolve():
        raise ValueError("RunName must identify a direct child of baseline/runs")
    return result


def check_config() -> dict:
    from dotenv import dotenv_values, load_dotenv
    if not ENV_FILE.is_file():
        raise ValueError(f"找不到配置文件：{ENV_FILE}")
    configured = dotenv_values(ENV_FILE)
    missing = [name for name in ("GLM_API_KEY", "GLM_BASE_URL", "GLM_MODEL")
               if not str(configured.get(name) or "").strip()]
    if missing:
        raise ValueError(".env缺少配置：" + "、".join(missing))
    load_dotenv(ENV_FILE, override=True)
    endpoint = urlparse(os.environ["GLM_BASE_URL"])
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("GLM_BASE_URL不是有效的HTTP/HTTPS接口地址。")
    from baseline.scripts.prompt_data_sources import training_identity, require_current_split
    identity = training_identity()
    require_current_split(identity)
    pilot = json.loads(PLAN.read_text(encoding="utf-8"))
    if pilot.get("source_sha256") != identity["source_sha256"]:
        raise ValueError("固定十首整曲名单不对应当前训练文件，请重新核验新版划分。")
    if pilot.get("score_count") != 10 or pilot.get("prefix_phrases") != 0:
        raise ValueError("自动训练只接受当前冻结的十首整曲清单（score_count=10, prefix_phrases=0）。")
    return {"env_file": str(ENV_FILE), "required_variables": {name: "configured" for name in
            ("GLM_API_KEY", "GLM_BASE_URL", "GLM_MODEL")}, "pilot_scores": pilot["score_count"],
            "pilot_rows": {name: part["rows"] for name, part in pilot["splits"].items()},
            "source_sha256": identity["source_sha256"]}


def check_api():
    from baseline.gepa_guqin.glm_client import GlmAnthropicClient, visible_text
    client = GlmAnthropicClient(max_retries=0, timeout=45)
    response = client.messages_create(
        model=client.model, max_tokens=32, temperature=0,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": "Reply with OK."}],
    )
    if not visible_text(response):
        raise RuntimeError("API返回了响应，但没有可读取的文字。请检查模型和接口协议。")
    usage = getattr(response, "usage", None)
    return {"status": "api_connected", "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0)}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        from ctypes import wintypes
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        dll.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        dll.OpenProcess.restype = wintypes.HANDLE
        handle = dll.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: do not assume dead.
        try:
            code = wintypes.DWORD()
            dll.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            dll.CloseHandle.argtypes = [wintypes.HANDLE]
            return bool(dll.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            dll.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class RunLock:
    def __init__(self, root):
        self.path = root / "auto_training.lock"

    def __enter__(self):
        if self.path.exists():
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            if previous.get("schema") != "gepa-auto-lock-1":
                raise RuntimeError("运行锁格式未知，请检查运行目录。")
            if process_alive(int(previous.get("pid", -1))):
                raise RuntimeError("同一实验已经在运行。请查看网页进度，不要重复启动。")
            self.path.unlink()  # Only this script's known stale lock file.
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump({"schema": "gepa-auto-lock-1", "pid": os.getpid(), "created_at": timestamp()}, handle)
        return self

    def __exit__(self, *args):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                self.path.unlink()


def snapshot(root: Path) -> dict:
    records = []
    events = root / "teacher_gepa_events.jsonl"
    if events.exists():
        with events.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
    scored = [r for r in records if r.get("event") == "teacher_phrase_scored"]
    proposals = [r for r in records if r.get("event") == "gepa_proposal"]
    confirmation_records = 0
    confirmation_path = root / "confirmation/rollout_events.jsonl"
    if confirmation_path.exists():
        with confirmation_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    confirmation_records += int(json.loads(line).get("event") == "teacher_phrase_scored")
                except ValueError:
                    pass
    return {"scored_records": len(scored), "proposals_recorded": len(proposals),
            "confirmation_records": confirmation_records,
            "last_sample": scored[-1].get("sample_id") if scored else None,
            "last_event_at": records[-1].get("logged_at") if records else None}


def update(root: Path, status: dict):
    payload = {**status, "updated_at": timestamp(), "training_progress": snapshot(root)}
    temporary = root / "auto_training_status.json.tmp"
    temporary.write_text(redact(json.dumps(payload, ensure_ascii=False, indent=2)) + "\n", encoding="utf-8")
    temporary.replace(root / "auto_training_status.json")
    labels = {"running": "正在运行", "prepared": "准备完成，尚未调用API训练", "complete": "本轮已结束",
              "failed": "任务报错，可查看日志后续跑", "interrupted": "已中断，可使用同一命令续跑"}
    state = labels.get(status.get("status"), status.get("status", ""))
    links = []
    for path, label in [("optimization_report.html", "查看每轮提示词、反馈、评分和逐格对照"),
                        ("confirmation/comparison.html", "查看独立确认结果"),
                        ("subprocess.log", "查看完整运行日志")]:
        if (root / path).exists():
            links.append(f"<li><a href='{path}'>{label}</a></li>")
    body = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
{'<meta http-equiv="refresh" content="10">' if status.get('status') == 'running' else ''}
<title>自动提示词训练进度</title><style>body{{font:16px/1.7 system-ui,'Microsoft YaHei';background:#17191c;color:#eee;max-width:1000px;margin:35px auto;padding:0 20px}}section{{background:#252a30;border:1px solid #46505b;border-radius:12px;padding:20px;margin:15px 0}}a{{color:#9bd0ff}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}small{{color:#bbb}}</style>
<h1>自动提示词训练</h1><section><h2>{html.escape(state)}</h2>
<p>当前阶段：{html.escape(str(status.get('stage', '检查配置')))}</p>
<p>{html.escape(redact(str(status.get('message', ''))))}</p>
<p>已记录提案：{payload['training_progress']['proposals_recorded']}；评测记录：{payload['training_progress']['scored_records']}</p>
<p>独立确认评测记录：{payload['training_progress']['confirmation_records']}</p>
<p>最近样本：{html.escape(str(payload['training_progress']['last_sample'] or '等待首条结果'))}</p>
<small>更新时间（UTC）：{payload['updated_at']}。运行时每10秒刷新；评测记录包含父代及候选，不能当作独立样本数。</small></section>
<section><h2>报告与日志</h2><ul>{''.join(links)}</ul></section></html>"""
    (root / "auto_training.html").write_text(body, encoding="utf-8", newline="\n")


def run_stage(root: Path, status: dict, name: str, arguments: list[str]):
    status.update(status="running", stage=name, message="正在执行，结果会自动写入当前实验目录。")
    update(root, status)
    print(f"[{name}] 开始。网页进度：{root / 'auto_training.html'}", flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE / ".gepa_packages"), str(REPO), str(REPO / "baseline/gepa/src")])
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    child = subprocess.Popen([sys.executable, *arguments], cwd=WORKSPACE, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
    messages = queue.Queue()

    def collect():
        for line in child.stdout:
            messages.put(line)
        messages.put(None)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    last_update = 0.0
    last_console = time.monotonic()
    ended = False
    try:
        with (root / "subprocess.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n[{timestamp()}] {name}\n")
            while not ended:
                try:
                    line = messages.get(timeout=1)
                    if line is None:
                        ended = True
                    else:
                        log.write(redact(line))
                        log.flush()
                except queue.Empty:
                    pass
                if time.monotonic() - last_update >= 10:
                    update(root, status)
                    last_update = time.monotonic()
                if time.monotonic() - last_console >= 30:
                    progress = snapshot(root)
                    print(f"[{name}] 仍在运行；已记录提案 {progress['proposals_recorded']}，"
                          f"训练评测 {progress['scored_records']}，独立确认 {progress['confirmation_records']}。", flush=True)
                    last_console = time.monotonic()
            code = child.wait()
    except KeyboardInterrupt:
        # Ctrl+C normally reaches both processes. Give the child time to save
        # its GEPA state before terminating this task-owned subprocess.
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=8)
        raise
    if code:
        raise RuntimeError(f"阶段“{name}”退出码为{code}。原因见 {root / 'subprocess.log'}；修正后执行同一命令续跑。")
    print(f"[{name}] 完成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-prompt", choices=("production",), default="production")
    parser.add_argument("--series", default="M")
    parser.add_argument("--run-name")
    parser.add_argument("--max-metric-calls", type=int, default=60)
    parser.add_argument("--max-proposals", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=2)
    parser.add_argument("--confirmation-repeats", type=int, default=2)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-config", action="store_true", help="offline validation only")
    parser.add_argument("--check-api", action="store_true", help="one small connectivity request, without training")
    parser.add_argument("--score-key", help="one-score smoke test, chosen only from iteration_optimize")
    parser.add_argument("--feedback-phrases", type=int, default=2)
    parser.add_argument("--pareto-phrases", type=int, default=1)
    parser.add_argument("--feedback-detail", choices=("exact", "features"), default="exact")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Z]", args.series):
        parser.error("--series必须是一个大写字母，新十首整曲系列默认为M。")
    args.run_name = args.run_name or (f"{args.series}_single_{args.score_key}_from_original" if args.score_key
                                    else f"{args.series}_whole10_from_original")
    root = run_path(args.run_name)
    configuration = check_config()
    target = f"单曲谱 {args.score_key}" if args.score_key else "固定随机十首整曲划分"
    print(f"配置检查通过：已从 guqinagent/.env 读取GLM配置（不显示密钥）；{target}。", flush=True)
    if args.check_api:
        print(json.dumps(check_api(), ensure_ascii=False))
        return 0
    if args.check_config:
        print(json.dumps(configuration, ensure_ascii=False, indent=2))
        return 0
    if min(args.max_metric_calls, args.max_proposals, args.minibatch_size, args.confirmation_repeats) < 1:
        parser.error("预算、提案数、小批大小和确认重复数都必须为正数。")
    root.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "stage": "检查固定名单", "started_at": timestamp(),
              "run_name": args.run_name, "seed_prompt": args.seed_prompt, "series": args.series}
    with RunLock(root):
        try:
            run_stage(root, status, "核验新版数据划分", [str(SCRIPTS / "create_new_dataset_gepa_split.py"), "--check"])
            arguments = [str(SCRIPTS / "run_teacher_gepa_iteration.py"),
                         "--run-dir", str(root), "--seed-prompt", args.seed_prompt,
                         "--max-metric-calls", str(args.max_metric_calls), "--max-proposals", str(args.max_proposals),
                         "--minibatch-size", str(args.minibatch_size), "--seed", "20261029",
                         "--max-rounds", "8", "--attempts", "1", "--temperature", "0",
                         "--feedback-detail", args.feedback_detail, "--series", args.series]
            if args.score_key:
                arguments += ["--score-key", args.score_key, "--feedback-phrases", str(args.feedback_phrases),
                              "--pareto-phrases", str(args.pareto_phrases)]
            else:
                arguments += ["--pilot-manifest", str(PLAN)]
            if args.prepare_only:
                arguments.append("--prepare-only")
            run_stage(root, status, "准备训练" if args.prepare_only else "GLM自动迭代训练", arguments)
            if args.prepare_only:
                status.update(status="prepared", message="准备检查已完成；去掉 -PrepareOnly 即可开始真实API训练。")
            else:
                selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
                if selection["safe_best_is_seed"]:
                    status.update(status="complete", message="本轮没有候选通过三指标安全门，保留种子提示词，未额外运行确认集。")
                elif args.skip_confirmation or args.score_key:
                    status.update(status="complete", message=(
                        "单曲谱测试已完成；结果只验证自动训练和标注差异流程，不能宣布跨曲谱提升。"
                        if args.score_key else "候选已通过初筛；按参数跳过独立确认，尚不能宣布正式胜出。"))
                else:
                    run_stage(root, status, "候选与种子独立确认", [str(SCRIPTS / "evaluate_new_data_gepa.py"),
                              "--run-dir", str(root), "--mode", "confirmation", "--repeats", str(args.confirmation_repeats)])
                    confirmation = json.loads((root / "confirmation/paired_result.json").read_text(encoding="utf-8"))
                    status.update(status="complete", message=(
                        "候选通过初步独立确认；请查看报告，最终test50仍需单独执行。"
                        if confirmation.get("provisional_safety_pass") else
                        "候选未通过独立确认，保留种子；本轮结果已归档。"))
            update(root, status)
        except KeyboardInterrupt:
            status.update(status="interrupted", message="已中断。执行同一命令恢复最近检查点；进行中的调用可能需要重做。")
            update(root, status)
            print(status["message"], flush=True)
            return 130
        except Exception as exc:
            status.update(status="failed", message=redact(str(exc)))
            update(root, status)
            raise
    print(status["message"], flush=True)
    print(f"进度页：{root / 'auto_training.html'}", flush=True)
    print(f"详细报告：{root / 'optimization_report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("运行失败：" + redact(str(exc)), file=sys.stderr, flush=True)
        raise SystemExit(1)
