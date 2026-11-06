#!/usr/bin/env python3
"""test50 最终测试：N0/N55/N63/N120 四臂对比（用户显式授权开启封存集）。

- 复用 evaluate_new_data_gepa.py 的评测协议（同模型、同adapter、官方评分器）
- 每臂独立断点续跑，进度实时写 test50_top3/status.json
- 完成后渲染含音高核对（✓/✗）的 HTML 对比报告
"""
from __future__ import annotations

import hashlib
import html as html_mod
import json
import os
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "baseline/runs/N_gepa16_from_original"
OUT = RUN / "test50_top3"
sys.path[:0] = [str(REPO.parent / ".gepa_packages"), str(REPO),
                str(REPO / "baseline/gepa/src")]

from baseline.gepa_guqin.adapter import USER_PROMPT_CONTEXT_VERSION
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.metric import replay_events
from baseline.scripts.prompt_data_sources import TEST50
from baseline.scripts.run_teacher_gepa_iteration import JsonlLogger, seed_prompt
from train.scripts.score_agent_predictions import score as official_score

ARMS = ["N0", "N55", "N63", "N120"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_prompts():
    manifest = json.loads((RUN / "experiment_manifest.json").read_text(encoding="utf-8"))
    seed_text, _ = seed_prompt(manifest["parameters"]["seed_prompt"])
    sel = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    cands = json.loads((RUN / "candidates.json").read_text(encoding="utf-8"))
    texts = {"N0": seed_text}
    for version in ARMS[1:]:
        entry = next(c for c in sel["official_candidate_scores"] if c["version"] == version)
        texts[version] = cands[entry["candidate_idx"]]["single_stage"]
    return texts, manifest


def _pid_alive(pid: int) -> bool:
    """Windows 兼容的存活检查：os.kill(pid,0) 在 Windows 上语义不对，改用 OpenProcess。"""
    import ctypes
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    try:
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    except Exception:
        return False
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == 259  # STILL_ACTIVE
        return False
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def acquire_lock(path):
    """进程锁：防止两个监督器同时跑同一臂写同一断点文件。"""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("pid", -1))
        if _pid_alive(pid):
            raise SystemExit(f"另一个实例（PID {pid}）正在跑，退出以免覆盖断点。")
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "arm": path.stem}, handle, ensure_ascii=False)
    return path


def release_lock(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("pid", -1)) == os.getpid():
            path.unlink()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, help="只跑指定臂（用于并行）")
    parser.add_argument("--finalize", action="store_true", help="合并四臂并打官方分")
    parser.add_argument("--recover-stale-lock", action="store_true",
                        help="归档已失效的本地锁，从断点恢复指定臂")
    parser.add_argument("--workers", type=int, default=1,
                        help="曲谱级并发数；同一曲谱内始终串行，默认1")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers 必须在1到8之间")
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=True)
    texts, manifest = load_prompts()
    public = read_jsonl(TEST50 / "test_runtime.jsonl")
    references = {x["sample_id"]: x for x in read_jsonl(TEST50 / "evaluation_pairs_test.jsonl")}
    examples = []
    for row in public:
        ref = references.get(row["sample_id"], {})
        examples.append({"sample_id": row["sample_id"], "score_key": row["score_key"],
                         "phrase_id": row["phrase_id"], "runtime_item": row["runtime_item"],
                         "reference": ref.get("reference", {"actions": []})})
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        return finalize(texts)
    lock_path = OUT / f"lock_{args.arm}.json"
    if args.recover_stale_lock and lock_path.exists():
        stale = json.loads(lock_path.read_text(encoding="utf-8"))
        if _pid_alive(int(stale.get("pid", -1))):
            raise SystemExit("锁对应的评测进程仍在运行；拒绝重复启动。")
        archived = lock_path.with_name(f"{lock_path.stem}.stale_{int(time.time())}.json")
        lock_path.replace(archived)
        print(f"已归档失效锁：{archived.name}；从断点恢复。", flush=True)
    lock = acquire_lock(lock_path)
    try:
        return run_arm(args.arm, texts, manifest, examples, workers=args.workers)
    finally:
        release_lock(lock)


def run_arm(run_arm, texts, manifest, examples, *, workers=1):
    arm_path = OUT / f"rows_{run_arm}.json"
    configuration = {
        "mode": "test50_top3_user_authorized",
        "user_prompt_context_version": USER_PROMPT_CONTEXT_VERSION,
        "sample_count": len(examples),
        "sample_ids": [e["sample_id"] for e in examples],
        "prompt_hashes": {k: hashlib.sha256(compose_system(v).encode()).hexdigest()
                          for k, v in texts.items()},
        "parameters": manifest["parameters"],
        "model": os.environ.get("GLM_MODEL", "glm-5.3"),
        "max_tokens": os.environ.get("GLM_MAX_TOKENS", "8000"),
        "thinking": os.environ.get("GLM_THINKING", "disabled"),
        "public_sha256": hashlib.sha256((TEST50 / "test_runtime.jsonl").read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((TEST50 / "evaluation_pairs_test.jsonl").read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256((REPO / "agents/ToolRuntime/runtime.py").read_bytes()).hexdigest(),
    }
    result = json.loads(arm_path.read_text(encoding="utf-8")) if arm_path.exists() else {
        "configuration": configuration, "rows": [], "status": "running", "arm": run_arm,
    }
    if result["configuration"] != configuration or result["arm"] != run_arm:
        raise SystemExit("Checkpoint protocol differs. Archive the old arm output before starting a fresh run.")
    completed = {(r["sample_id"]): r for r in result["rows"]}
    examples_by_id = {str(example["sample_id"]): example for example in examples}
    historical = {}
    comparison_rows = []
    for example in examples:
        item = deepcopy(example["runtime_item"])
        item["input"].setdefault("phrase_handoff", {}).pop("previous_phrase", None)
        row = {"trajectory_id": example["sample_id"], "score_key": example["score_key"],
               "phrase_id": example["phrase_id"], "input": item["input"],
               "baseline_plan": item.get("baseline_plan") or {},
               "reference_plan": {"actions": materialize_actions(example["reference"]["actions"])}}
        historical[(example["score_key"], example["phrase_id"])] = row
        comparison_rows.append(row)
    with (OUT / "comparison_dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in comparison_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "references_for_scorer.jsonl").open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps({"sample_id": example["sample_id"],
                                 "reference": example["reference"]}, ensure_ascii=False) + "\n")

    total = len(examples)
    state_lock = threading.Lock()
    # The dashboard is deliberately rendered by this long-lived parent process,
    # rather than by a separate watcher.  That makes the local HTML page survive
    # a terminal/session restart and is safe with several score workers: each
    # worker first atomically replaces rows_N55.json, then asks this one thread
    # to refresh at most once per 15 seconds.
    dashboard_stop = threading.Event()
    dashboard_signal = threading.Event()
    dashboard_thread = None

    def refresh_dashboard():
        if run_arm != "N55":
            return
        try:
            from baseline.scripts.render_test50_n55_live import render
            render()
        except Exception as exc:
            # Reporting must never interrupt an expensive sealed evaluation.
            print(f"N55 实时HTML刷新失败（评测继续）：{type(exc).__name__}: {exc}", flush=True)

    def dashboard_loop():
        while True:
            dashboard_signal.wait(15)
            dashboard_signal.clear()
            refresh_dashboard()
            if dashboard_stop.is_set():
                return

    if run_arm == "N55":
        dashboard_thread = threading.Thread(target=dashboard_loop, name="N55-dashboard", daemon=True)
        dashboard_thread.start()
        dashboard_signal.set()
    grouped = defaultdict(list)
    for example in examples:
        if example["sample_id"] not in completed:
            grouped[example["score_key"]].append(example)
    print(f"test50 单臂评测：{total}样本 × {run_arm}；曲谱级并发={workers}，剩余{sum(map(len, grouped.values()))}句", flush=True)

    def run_score(score_key, score_examples, worker_slot):
        client = GlmAnthropicClient(min_interval=0.5)
        log = JsonlLogger(OUT / f"rollout_events_{run_arm}_worker{worker_slot}.jsonl")
        adapter = TeacherExactGuqinAdapter(
            examples, historical, GlmBackend(client),
            max_rounds=int(manifest["parameters"]["max_rounds"]),
            attempts=int(manifest["parameters"]["attempts"]),
            sample_temperature=float(manifest["parameters"]["temperature"]), log=log)
        cache_key = hashlib.sha256(compose_system(texts[run_arm]).encode()).hexdigest()[:16]
        sessions = adapter._session_cache.setdefault(cache_key, {})
        # Each worker owns one score at a time; only that score's saved prefix
        # is restored, so phrase handoff remains serial and candidate-owned.
        with state_lock:
            prior = {sid: row for sid, row in completed.items()
                     if examples_by_id.get(sid, {}).get("score_key") == score_key}
        for sid, row in prior.items():
            sessions[sid] = {"ok": row["protocol_ok"], "trace": [],
                             "plan": {"actions": [{"source_index": int(i), "jianzi_text": text}
                                                  for i, text in (row.get("prediction") or {}).get("jianzi_rows", [])]},
                             "error": row.get("error")}
        for example in sorted(score_examples, key=lambda item: item["phrase_id"]):
            output = None
            for attempt in range(3):
                try:
                    output = adapter.evaluate([example], {"single_stage": texts[run_arm]}).outputs[0]
                    if output["protocol_ok"] or attempt == 2:
                        break
                    failed = [sid for sid, session in sessions.items() if not session.get("ok")]
                    for sid in failed:
                        sessions.pop(sid, None)
                    print(f"retry {run_arm}/{example['sample_id']}：重建 {len(failed)} 个失败前序会话", flush=True)
                    time.sleep(2)
                except Exception as exc:
                    print(f"retry {run_arm}/{example['sample_id']}: {exc}", flush=True)
                    time.sleep(30)
            if output is None:
                raise RuntimeError(f"{score_key} 在三次API尝试后仍未获得输出")
            record = {"sample_id": example["sample_id"], "score_key": example["score_key"],
                      "phrase_id": example["phrase_id"], "arm": run_arm,
                      "protocol_ok": output["protocol_ok"], "teacher_score": output["teacher_score"],
                      "prediction": {"jianzi_rows": output["jianzi_rows"]},
                      "error": None if output["protocol_ok"] else "rollout_or_predecessor_failure"}
            with state_lock:
                result["rows"].append(record)
                completed[record["sample_id"]] = record
                save(arm_path, result)
                done = len(result["rows"])
                dashboard_signal.set()
            print(f"[{run_arm} {done}/{total}｜worker{worker_slot}] {record['sample_id']} ok={record['protocol_ok']}", flush=True)

    try:
        if grouped:
            # A task is a whole score, never an individual phrase.  Therefore
            # no two threads can race on phrase_handoff / predecessor state.
            ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{run_arm}-score") as pool:
                futures = [pool.submit(run_score, key, values, position % workers + 1)
                           for position, (key, values) in enumerate(ordered)]
                for future in as_completed(futures):
                    future.result()
        if len(completed) < total:
            result["status"] = "arm_incomplete"
            save(arm_path, result)
            return 1
        result["status"] = "arm_complete"
        save(arm_path, result)
        print(f"{run_arm} 臂完成", flush=True)
        return 0
    finally:
        if dashboard_thread is not None:
            dashboard_stop.set()
            dashboard_signal.set()
            dashboard_thread.join(timeout=60)


def finalize(texts):
    """已完成的臂合并、官方评分、写总结果（只收 status=arm_complete 的臂）。"""
    refs = read_jsonl(OUT / "references_for_scorer.jsonl")
    refs_by_id = {r["sample_id"]: r for r in refs}
    official = {}
    merged_rows = []
    done_arms = []
    for arm in ARMS:
        arm_file = OUT / f"rows_{arm}.json"
        if not arm_file.exists():
            continue
        arm_data = json.loads(arm_file.read_text(encoding="utf-8"))
        if arm_data.get("status") != "arm_complete":
            print(f"跳过未完成臂 {arm}（status={arm_data.get('status')}）", flush=True)
            continue
        done_arms.append(arm)
        merged_rows.extend(arm_data["rows"])
        preds_path = OUT / f"{arm}_predictions.jsonl"
        with preds_path.open("w", encoding="utf-8") as fh:
            for row in arm_data["rows"]:
                fh.write(json.dumps({"sample_id": row["sample_id"], "protocol_valid": row["protocol_ok"],
                                     "jianzi_rows": row["prediction"]["jianzi_rows"]}, ensure_ascii=False) + "\n")
        scores_path = OUT / f"{arm}_official_score.json"
        official_score(SimpleNamespace(reference=OUT / "references_for_scorer.jsonl",
                                       predictions=preds_path, output=scores_path))
        official[arm] = json.loads(scores_path.read_text(encoding="utf-8"))["aggregate"]
    if not done_arms:
        print("没有任何完成的臂，无法 finalize", flush=True)
        return 1
    save(OUT / "paired_result.json", {
        "status": "complete", "arms": done_arms,
        "configuration": {"mode": "test50_top3_user_authorized",
                          "prompt_hashes": {k: hashlib.sha256(compose_system(v).encode()).hexdigest()
                                            for k, v in texts.items()}},
        "rows": merged_rows, "official_scores": official})
    print(json.dumps({"status": "complete", "official_scores": official}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
