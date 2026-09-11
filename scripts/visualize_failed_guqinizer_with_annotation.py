#!/usr/bin/env python3
"""Create a private diagnostic page for failed Guqinizer calls with base/annotation rows."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def esc(v: object) -> str:
    return html.escape(str(v if v is not None else ""), quote=True)


def strip_brackets(v: str) -> str:
    v = (v or "").strip()
    return v[1:-1] if v.startswith("[") and v.endswith("]") else v


def parse_table(text: str, *, start_marker: str | None = None) -> dict[int, dict[str, str]]:
    if start_marker and start_marker in text:
        text = text.split(start_marker, 1)[1]
    rows: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        cells = line.split("｜")
        if len(cells) < 4 or not cells[0].strip().isdigit():
            continue
        index = int(cells[0].strip())
        jianzi = cells[-1].strip() if len(cells) >= 5 else ""
        rows[index] = {
            "jianpu": cells[1].strip(),
            "abc": cells[2].strip(),
            "duration": cells[3].strip(),
            "jianzi": strip_brackets(jianzi),
        }
    return rows


def parse_annotation(system: str) -> dict[int, dict[str, str]]:
    marker = "【教师私有参考谱】"
    if marker not in system:
        return {}
    return parse_table(system.split(marker, 1)[1])


def response_text(response: dict) -> str:
    content = response.get("content")
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--inferred", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    failures: dict[str, dict] = {}
    for report_path in sorted(args.shards.glob("shard_*/generation_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for failure in report.get("failures", []):
            failures[failure["trajectory_id"]] = failure
    ids = set(failures)

    inferred: dict[str, dict] = {}
    with args.inferred.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("trajectory_id") in ids:
                inferred[item["trajectory_id"]] = item

    # teacher_rejected_io contains first-round and retry records. Keep the last
    # five records for each ID, which is the final retry round.
    rejected_records: dict[str, list[dict]] = {}
    for path in sorted(args.shards.glob("shard_*/teacher_rejected_io.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                tid = item.get("trajectory_id")
                if tid in ids:
                    rejected_records.setdefault(tid, []).append(item)

    articles: list[str] = []
    for tid in sorted(ids):
        failure = failures[tid]
        records = rejected_records.get(tid, [])
        records = records[-5:] if len(records) > 5 else records
        records.sort(key=lambda x: int(x.get("attempt", 0)))
        item = records[-1] if records else {}
        trace = item.get("teacher_io_trace") or []
        first_req = (trace[0].get("request") or {}) if trace else {}
        user = ((first_req.get("messages") or [{}])[0]).get("content", "")
        current = parse_table(user.split("【当前段", 1)[1] if "【当前段" in user else user)
        annotation = parse_annotation(first_req.get("system", ""))
        # This current-segment table is exactly the Base/Fingering output sent
        # to Guqinizer, so it is more faithful than the inferred baseline plan.
        base = {i: row.get("jianzi", "") for i, row in current.items()}
        indices = sorted(set(current) | set(annotation) | set(base))
        nonempty_ann = sum(bool(annotation.get(i, {}).get("jianzi")) for i in indices)
        nonempty_base = sum(bool(base.get(i, "")) for i in indices)
        rows = []
        for i in indices:
            c = current.get(i, {})
            ann = annotation.get(i, {}).get("jianzi", "")
            b = base.get(i, "")
            ann_display = "［空］" if not ann or ann == "空" else ann
            base_display = "［空］" if not b else b
            rows.append(
                f"<tr><td>{i}</td><td>{esc(c.get('jianpu') or annotation.get(i, {}).get('jianpu',''))}</td>"
                f"<td>{esc(base_display)}</td><td>{esc(ann_display)}</td></tr>"
            )
        attempts = []
        for n, record in enumerate(records, 1):
            call = (record.get("teacher_io_trace") or [{}])[-1]
            attempts.append(
                f"<details><summary>第 {n} 次响应（API attempt {record.get('attempt','?')}）</summary>"
                f"<p class=error>{esc(record.get('error',''))}</p><pre>{esc(response_text(call.get('response') or {}))}</pre></details>"
            )
        articles.append(
            f"<article id='{esc(tid)}'><h2>{esc(tid)}</h2>"
            f"<p class=meta>阶段：Guqinizer｜最终失败：{esc(failure.get('error',''))}｜最终重试轮次：{len(records)} 次｜"
            f"当前行：{len(current)}，标注非空：{nonempty_ann}，Base 非空：{nonempty_base}</p>"
            f"<table><thead><tr><th>序号</th><th>简谱</th><th>Base 当前减字</th><th>标注减字</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><h3>模型失败响应</h3>{''.join(attempts)}</article>"
        )

    page = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Guqinizer 失败样本诊断</title>
<style>body{margin:0;background:#f5f3ee;color:#28251f;font:14px/1.55 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1200px;margin:auto;padding:24px 18px}h1{margin:0 0 6px}h2{margin:0 0 8px;font-size:18px}h3{font-size:15px;margin:16px 0 6px}.intro{color:#6d675e;margin-bottom:18px}.nav{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 22px}.nav a{color:#175b70;text-decoration:none}.sample,article{background:#fffdf8;border:1px solid #d9d2c4;border-radius:10px;padding:16px;margin:0 0 18px;overflow:hidden}.meta{color:#6d675e;font-size:12px;overflow-wrap:anywhere}.error{color:#9a3d24;white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0 12px}th,td{border:1px solid #ddd5c7;padding:6px 7px;vertical-align:top;text-align:left}th{background:#eee9df}td:nth-child(3),td:nth-child(4){max-width:330px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f4ed;padding:10px;margin:0}details{border:1px solid #ddd5c7;margin:6px 0;padding:5px}summary{cursor:pointer;font-weight:600}</style></head><body><main>
<h1>Guqinizer 82 条失败样本诊断</h1><p class='intro'>每条显示 Base 当前减字与私有标注逐序号对照，并保留模型各次失败响应。此页包含私有标注，仅用于人工排查，不得直接用于训练。</p>
""" + "<nav class='nav'>" + "".join(f"<a href='#{esc(t)}'>{esc(t)}</a>" for t in sorted(ids)) + "</nav>" + "".join(articles) + "</main></body></html>"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(articles), "missing_trace": len(ids - set(rejected_records))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
