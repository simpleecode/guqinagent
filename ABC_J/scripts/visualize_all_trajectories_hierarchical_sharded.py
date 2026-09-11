#!/usr/bin/env python3
"""Split the hierarchical trajectory viewer into one HTML file per score."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_key_from_message(row: dict) -> str:
    provenance = row.get("provenance") or {}
    value = str(provenance.get("score_key") or provenance.get("source_trajectory_id") or "")
    return value.split("-p", 1)[0]


def score_key_from_source(row: dict) -> str:
    return str(row.get("score_key") or row.get("score_family_id") or row.get("trajectory_id") or "").split("-p", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    messages_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.input):
        messages_by_score[score_key_from_message(row)].append(row)
    audits = {row["sample_id"]: row for row in read_jsonl(args.audit)}
    sources_by_score: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.source):
        sources_by_score[score_key_from_source(row)].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    viewer = Path(__file__).with_name("visualize_all_trajectories_hierarchical.py")
    entries = []
    with tempfile.TemporaryDirectory(prefix="guqin-hierarchical-") as temp_name:
        temp = Path(temp_name)
        for position, score_key in enumerate(sorted(messages_by_score), 1):
            messages = messages_by_score[score_key]
            sample_ids = {row["sample_id"] for row in messages}
            score_audits = [audits[sample_id] for sample_id in sample_ids if sample_id in audits]
            score_sources = sources_by_score.get(score_key, [])
            input_path = temp / "messages.jsonl"
            audit_path = temp / "audit.jsonl"
            source_path = temp / "source.jsonl"
            write_jsonl(input_path, messages)
            write_jsonl(audit_path, score_audits)
            write_jsonl(source_path, score_sources)
            output_name = f"{score_key}.html"
            subprocess.run([
                sys.executable, str(viewer),
                "--input", str(input_path), "--audit", str(audit_path),
                "--source", str(source_path), "--output", str(args.output_dir / output_name),
            ], check=True, stdout=subprocess.DEVNULL)
            phrase_count = len({(row.get("provenance") or {}).get("source_trajectory_id") for row in messages})
            title = next((str(row.get("title") or "") for row in score_sources if row.get("title")), "")
            entries.append((score_key, title, phrase_count, output_name))
            print(f"[{position}/{len(messages_by_score)}] {score_key}: {phrase_count}", flush=True)

    links = "\n".join(
        f'<li><a href="{html.escape(filename)}"><code>{html.escape(score_key)}</code>'
        f'{"｜" + html.escape(title) if title else ""}</a><span>{count} phrases</span></li>'
        for score_key, title, count, filename in entries
    )
    index = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>古琴 Agent 全量轨迹索引</title>
<style>body{{margin:0;background:#f6f7f9;color:#202124;font:15px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:920px;margin:auto;padding:28px}}h1{{font-size:24px}}p{{color:#666}}ul{{list-style:none;padding:0;background:#fff;border-radius:10px;overflow:hidden}}li{{display:flex;justify-content:space-between;gap:16px;padding:10px 14px;border-bottom:1px solid #eee}}li:last-child{{border:0}}a{{color:#175cd3;text-decoration:none}}span{{color:#777;white-space:nowrap}}@media(prefers-color-scheme:dark){{body{{background:#17191c;color:#eee}}p,span{{color:#aaa}}ul{{background:#22252a}}li{{border-color:#34383e}}a{{color:#83b4ff}}}}</style></head><body><main><h1>古琴 Agent 全量轨迹</h1>
<p>{len(entries)} 首曲目；按曲目打开，避免一次加载全量轨迹。</p><ul>{links}</ul></main></body></html>'''
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "index.html"), "scores": len(entries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
