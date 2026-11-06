#!/usr/bin/env python3
"""Resume this fresh-series experiment in batches until provisional improvement.

The confirmation partition is a repeated screening holdout in this controller;
it is not the sealed final test. Its labels and errors never enter reflection.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(WORKSPACE_ROOT / ".gepa_packages"), str(WORKSPACE_ROOT / "guqinagent"),
               str(WORKSPACE_ROOT / "guqinagent/baseline/gepa/src")]
from baseline.scripts.run_auto_prompt_training import (
    RunLock, SCRIPTS, WORKSPACE, check_config, process_alive, redact, run_path, timestamp,
)


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def confirmed_improvement(root):
    selection = read(root / "selection.json", {})
    best = read(root / "best_candidate.json", {})
    result = read(root / "confirmation/paired_result.json", {})
    if selection.get("safe_best_is_seed", True) or result.get("status") != "complete":
        return False
    from baseline.gepa_guqin.handbook import compose_system
    text = best.get("safe_best_candidate", {}).get("single_stage")
    expected = hashlib.sha256(compose_system(text).encode()).hexdigest() if text else None
    if not expected or result.get("configuration", {}).get("prompt_hashes", {}).get("candidate") != expected:
        return False  # An earlier candidate's confirmation is never proof for a new one.
    means = result.get("repeat_mean_scores", {})
    baseline, candidate = means.get("baseline", {}), means.get("candidate", {})
    keys = ("exact_cell_rate", "nonempty_recall", "protocol_valid_rate")
    if any(baseline.get(k) is None or candidate.get(k) is None for k in keys):
        return False
    return (candidate["exact_cell_rate"] > baseline["exact_cell_rate"]
            and all(candidate[k] >= baseline[k] for k in keys))


def archive_confirmation(root):
    source = root / "confirmation"
    if not source.exists():
        return
    result = read(source / "paired_result.json", {})
    # Incomplete evaluations must be resumed, not discarded.
    if result.get("status") != "complete":
        raise RuntimeError("独立复核尚未完成；请先恢复原任务，不归档未完成的评测。")
    digest = result["configuration"]["prompt_hashes"]["candidate"][:16]
    target = root / "confirmation_history" / digest
    if not source.resolve().is_relative_to(root.resolve()) or not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("归档路径超出当前实验目录。")
    if target.exists():
        raise RuntimeError("该候选已有复核归档；保留两份原始记录并停止检查。")
    target.parent.mkdir(exist_ok=True)
    source.rename(target)


def confirmation_needed(root):
    selection = read(root / "selection.json", {})
    best = read(root / "best_candidate.json", {})
    text = best.get("safe_best_candidate", {}).get("single_stage")
    if selection.get("safe_best_is_seed", True) or not text:
        return False
    from baseline.gepa_guqin.handbook import compose_system
    digest = hashlib.sha256(compose_system(text).encode()).hexdigest()
    current = read(root / "confirmation/paired_result.json", {})
    current_hash = current.get("configuration", {}).get("prompt_hashes", {}).get("candidate")
    if current_hash == digest and current.get("status") == "complete":
        return False
    history = root / "confirmation_history" / digest[:16] / "paired_result.json"
    if history.exists():
        return False  # Never reroll the same candidate until it happens to pass.
    if current and current_hash != digest:
        archive_confirmation(root)
    return True


def publish(root, status, message, journal):
    payload = {"status": status, "message": message, "updated_at": timestamp(), "journal": journal,
               "success_scope": "pilot repeated screening, not sealed final-test superiority"}
    temporary = root / "continuous_iteration_status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "continuous_iteration_status.json")
    rows = "".join(f"<tr><td>{html.escape(str(e.get('at', '')))}</td>"
                   f"<td>{html.escape(str(e.get('message', '')))}</td></tr>" for e in journal)
    refresh = '<meta http-equiv="refresh" content="20">' if status == "running" else ""
    (root / "continuous_iteration.html").write_text(f"""<!doctype html><html lang="zh-CN">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>当前系列持续迭代</title><style>body{{font:16px/1.8 system-ui,'Microsoft YaHei';max-width:1100px;margin:35px auto;padding:0 20px;background:#f4f6fa;color:#263146}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dde3ec}}a{{color:#1565c0}}</style>
<h1>当前系列：从原版持续迭代</h1><section><h2>{html.escape(message)}</h2>
<p>原版为N0；续接当前候选和训练成败记录，不引入旧V系列。每批扩大搜索额度，评分标准与样本划分不变。</p>
<p>停止达标条件：验证筛选通过，且复核中逐格精确率提高、非空召回率与协议合规率均不下降。</p>
<p>复核集会用于多批次筛选，因此这里只能判定小规模筛选改善；最终test50保持封存，不能把本页当作最终测试结论。</p>
<a href="auto_training.html">本批实时进度</a> · <a href="optimization_report.html">每轮采样、修改、反馈与标注差异</a>
<p>更新时间（UTC）：{html.escape(payload['updated_at'])}</p></section>
<section><h2>续跑记录</h2><table><tr><th>时间（UTC）</th><th>操作</th></tr>{rows}</table></section></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="M_whole10_from_original")
    parser.add_argument("--proposal-step", type=int, default=6)
    parser.add_argument("--metric-step", type=int, default=60)
    parser.add_argument("--proposal-ceiling", type=int, default=0, help="0: user-authorized continuous search")
    args = parser.parse_args()
    if min(args.proposal_step, args.metric_step) <= 0 or args.proposal_ceiling < 0:
        parser.error("步长必须为正，额度上限不能为负。")
    check_config()
    root = run_path(args.run_name)
    if not (root / "experiment_manifest.json").exists():
        raise RuntimeError("请先建立并启动从原版开始的当前系列实验。")
    lock = RunLock(root)
    lock.path = root / "continuous_iteration.lock"
    previous = read(root / "continuous_iteration_status.json", {})
    journal = previous.get("journal", [])
    with lock:
        def note(message):
            journal.append({"at": timestamp(), "message": message})
            publish(root, "running", message, journal)
            print(message, flush=True)

        note("持续迭代控制已启动；等待当前批完成，避免重复调用。")
        try:
            while True:
                active = read(root / "auto_training.lock", {})
                if process_alive(int(active.get("pid", -1))):
                    publish(root, "running", "当前批正在运行，完成后自动检查并续跑。", journal)
                    time.sleep(20)
                    continue
                status = read(root / "auto_training_status.json", {})
                if status.get("status") != "complete":
                    raise RuntimeError("当前训练未正常完成；保留断点停止自动续跑，请查看本批日志。")
                if confirmed_improvement(root):
                    note("已通过小规模验证和复核改善门槛，停止自动搜索；最终测试尚未开启。")
                    publish(root, "complete", journal[-1]["message"], journal)
                    return 0
                if (root / "STOP_AUTO_ITERATION").exists():
                    publish(root, "stopped", "检测到停止标记，当前批已完成，不再启动下一批。", journal)
                    return 0
                manifest = read(root / "experiment_manifest.json", {})
                parameters = manifest["parameters"]
                series = manifest.get("series")
                if not isinstance(series, str) or len(series) != 1 or not series.isupper() or parameters["seed_prompt"] != "production":
                    raise RuntimeError("只允许续接从师兄库原版开始、使用单个大写字母编号的实验系列。")
                proposals = int(parameters["max_proposals"]) + args.proposal_step
                metrics = int(parameters["max_metric_calls"]) + args.metric_step
                if args.proposal_ceiling and proposals > args.proposal_ceiling:
                    publish(root, "limit_reached", "已达到配置的搜索额度，尚未证实优于原版。", journal)
                    return 0
                note(f"上一批未达完整改善门槛；续跑至累计{proposals}个提案、{metrics}次评测预算。")
                previous_rounds = len(read(root / "run_log.json", []))
                command = [sys.executable, str(SCRIPTS / "run_auto_prompt_training.py"),
                           "--run-name", args.run_name, "--seed-prompt", "production", "--series", series,
                           "--max-proposals", str(proposals), "--max-metric-calls", str(metrics),
                           "--minibatch-size", str(parameters["minibatch_size"]),
                           "--feedback-detail", str(parameters["feedback_detail"]), "--skip-confirmation"]
                with (root / "continuous_iteration.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while child.poll() is None:
                            publish(root, "running", f"自动续跑中：累计目标{proposals}个提案。", journal)
                            time.sleep(20)
                    except BaseException:
                        # The training runner owns its checkpoint; do not kill it mid-write.
                        note("续跑控制已退出；已启动的训练批继续保存结果，不会重复启动。")
                        raise
                if child.returncode:
                    raise RuntimeError(f"续跑退出码{child.returncode}，停止自动重试，请查看continuous_iteration.log。")
                if len(read(root / "run_log.json", [])) <= previous_rounds:
                    raise RuntimeError("扩大额度后没有生成新提案，停止空转，请检查GEPA停止原因。")
                if confirmation_needed(root):
                    note("新候选通过验证筛选，开始原版与该候选的双次复核；不重复抽测已失败的相同候选。")
                    with (root / "continuous_iteration.log").open("a", encoding="utf-8") as log:
                        confirmation = subprocess.Popen([
                            sys.executable, str(SCRIPTS / "evaluate_new_data_gepa.py"),
                            "--run-dir", str(root), "--mode", "confirmation", "--repeats", "2",
                        ], cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
                        while confirmation.poll() is None:
                            publish(root, "running", "正在进行候选与原版的双次复核。", journal)
                            time.sleep(20)
                    if confirmation.returncode:
                        raise RuntimeError("候选复核运行失败，停止自动重试并保留断点。")
        except Exception as exc:
            publish(root, "failed", redact(str(exc)), journal)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
