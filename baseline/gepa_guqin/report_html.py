"""Create an offline, auditable HTML report from a GEPA run directory.

Usage: python baseline/gepa_guqin/report_html.py --run-dir PATH [--output PATH]
The report shows recorded candidates, diffs, validation scores, iteration
decisions and rollout events. It cannot reconstruct hidden model reasoning or
candidate text that was not saved by the run.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def format_score(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "未记录"


def prompt_diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                 fromfile="parent", tofile="candidate", lineterm="")
    return esc("\n".join(lines) or "文本相同")


def candidate_rows(run_dir: Path, best: dict[str, Any]) -> list[dict[str, Any]]:
    saved = read_json(run_dir / "candidates.json", [])
    scores = best.get("val_aggregate_scores") or []
    if not isinstance(saved, list):
        saved = []
    count = max(len(saved), len(scores), int(best.get("num_candidates") or 0))
    if not count and best.get("best_candidate"):
        count = 1
    rows = []
    for index in range(count):
        candidate = saved[index] if index < len(saved) else None
        if candidate is None and index == best.get("best_idx"):
            candidate = best.get("best_candidate")
        prompt = candidate.get("single_stage") if isinstance(candidate, dict) else None
        rows.append({"index": index, "score": scores[index] if index < len(scores) else None,
                     "prompt": prompt, "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                     if isinstance(prompt, str) else None})
    return rows


def build_report(run_dir: Path) -> str:
    best = read_json(run_dir / "best_candidate.json", {})
    stats = read_json(run_dir / "api_stats.json", {})
    manifest = read_json(run_dir / "manifest.json", read_json(run_dir / "split_manifest.json", {}))
    iterations = read_json(run_dir / "run_log.json", [])
    text_log_path = run_dir / "gepa_run_log.txt"
    text_log = text_log_path.read_text(encoding="utf-8", errors="replace") if text_log_path.exists() else ""
    events = read_jsonl(run_dir / "adapter_log.jsonl")
    candidates = candidate_rows(run_dir, best)
    accepted = {int(idx): int(entry.get("selected_program_candidate") or 0)
                for entry in iterations for idx in (entry.get("new_program_indices") or [])
                if isinstance(idx, int)}
    cards = []
    for row in candidates:
        index, prompt, score = row["index"], row["prompt"], row["score"]
        parent = accepted.get(index)
        # Do not invent lineage when the run did not save iteration metadata.
        parent_text = candidates[parent]["prompt"] if parent is not None else None
        delta = score - candidates[parent]["score"] if (parent is not None
                and isinstance(score, (float, int))
                and isinstance(candidates[parent]["score"], (float, int))) else None
        details = (f'<details><summary>完整提示词</summary><pre>{esc(prompt)}</pre></details>'
                   f'<details><summary>相对父候选的文本变化</summary><pre>{prompt_diff(parent_text, prompt)}</pre></details>'
                   if isinstance(prompt, str) and isinstance(parent_text, str) else
                   f'<details><summary>已保存的提示词</summary><pre>{esc(prompt) if prompt else "本次运行未保存该候选文本"}</pre></details>')
        cards.append(
            f'<article><h3>候选 {index}{" · 验证集最佳" if index == best.get("best_idx") else ""}</h3>'
            f'<p>验证均分：{format_score(score)} · 父候选：{parent if parent is not None else "种子"}'
            f' · 相对变化：{format_score(delta)} · SHA-256：{esc(row["sha256"] or "未记录")}</p>{details}</article>')
    steps = []
    for entry in iterations if isinstance(iterations, list) else []:
        tasks = entry.get("tasks") or []
        before = entry.get("subsample_scores") or []
        after = entry.get("new_subsample_scores") or []
        decision = "生成候选 " + ", ".join(map(str, entry.get("new_program_indices") or [])) \
                   if entry.get("new_program_indices") else "该轮没有保存新候选"
        steps.append(
            f'<article><h3>第 {esc(entry.get("i", "?"))} 轮 · {esc(entry.get("iteration_id", ""))}</h3>'
            f'<p>父候选：{esc(entry.get("selected_program_candidate", "?"))} · {esc(decision)}'
            f' · 样本索引：{esc(entry.get("subsample_ids", []))}</p>'
            f'<p>原样本分：{esc(before)} → 新样本分：{esc(after)}</p>'
            f'<details><summary>该轮原始记录</summary><pre>{esc(json.dumps(tasks or entry, ensure_ascii=False, indent=2))}</pre></details></article>')
    event_rows = []
    for event in events:
        kind = event.get("event")
        if kind not in {"phrase_scored", "rollout_generation_error", "leakage_feedback"}:
            continue
        event_rows.append(f'<tr><td>{esc(kind)}</td><td>{esc(event.get("sample_id", ""))}</td>'
                          f'<td>{format_score(event.get("score"))}</td><td>{esc(event.get("ok", ""))}</td>'
                          f'<td>{esc(event.get("error", ""))}</td></tr>')
    explanation = ("评分来自已有运行记录；没有重新评测。候选选择以验证集记录为准，"
                   "新候选需经独立验证后才能替代生产提示词。页面展示可观察操作与日志，"
                   "不包含模型隐藏思维链。")
    baseline = read_json(run_dir / "pilot_vanilla_result.json", {})
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GEPA 提示词优化审计</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;background:#f4f7fa;color:#172238}}
article,section{{background:white;border:1px solid #dce3ea;border-radius:12px;padding:1rem 1.4rem;margin:1rem 0}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1rem;border-radius:8px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:.45rem;text-align:left}}
summary{{cursor:pointer;color:#174a8b}}small{{color:#567}}.scroll{{overflow:auto;max-height:550px}}</style></head><body>
<h1>GEPA 提示词优化审计</h1><p>{esc(explanation)}</p>
<section><h2>实验概况</h2><p>验证集最佳候选：{esc(best.get("best_idx", "未记录"))}；分数：{format_score(best.get("best_val_score"))}；候选数：{len(candidates)}</p>
<p>vanilla ReAct 历史分数：{format_score(baseline.get("mean_score"))}；该值只有同一运行目录存在原始记录时才展示。</p>
<details><summary>数据与用量记录</summary><pre>{esc(json.dumps({"manifest": manifest, "api_stats": stats}, ensure_ascii=False, indent=2))}</pre></details></section>
<section><h2>候选提示词与分数</h2>{''.join(cards) or '<p>未找到候选记录。</p>'}</section>
<section><h2>逐轮优化过程</h2>{''.join(steps) or '<p>本目录未保存逐轮 JSON；只能查看保存的文本日志。</p>'}
{f'<details><summary>原始 GEPA 文本日志</summary><pre>{esc(text_log)}</pre></details>' if text_log else ''}</section>
<section><h2>运行与失败记录</h2><div class="scroll"><table><thead><tr><th>事件</th><th>样本</th><th>得分</th><th>协议成功</th><th>错误</th></tr></thead><tbody>{''.join(event_rows)}</tbody></table></div>
<small>仅列出已保存的评测、生成失败与泄漏事件；原始记录仍在 run_dir 中。</small></section></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"run directory does not exist: {args.run_dir}")
    output = args.output or args.run_dir / "optimization_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(args.run_dir), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
