#!/usr/bin/env python3
"""Split the hierarchical trajectory viewer into one HTML file per score."""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    parser.add_argument("--notice", default="", help="Visible dataset provenance/compatibility notice")
    parser.add_argument("--token-lengths", type=Path,
                        help="Optional over-threshold TSV for indexed token-length labels")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    viewer = Path(__file__).with_name("visualize_all_trajectories_hierarchical.py")
    entries = []
    with tempfile.TemporaryDirectory(prefix="guqin-hierarchical-") as temp_name:
        temp = Path(temp_name)
        # Partition on disk; hold only one score's messages/source at a time.
        # Full-score source history is retained for stateful pitch replay.
        sample_scores = {}
        scores = set()
        for row in read_jsonl(args.input):
            score = score_key_from_message(row)
            if not score or Path(score).name != score:
                raise ValueError(f'Invalid score key: {score}')
            if row['sample_id'] in sample_scores:
                raise ValueError(f'Duplicate sample: {row["sample_id"]}')
            sample_scores[row['sample_id']] = score
            scores.add(score)
            with (temp/f'{score}.messages.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        for row in read_jsonl(args.source):
            score = score_key_from_source(row)
            if score in scores:
                with (temp/f'{score}.source.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        for row in read_jsonl(args.audit):
            score = sample_scores.get(row['sample_id'])
            if score is not None:
                with (temp / f'{score}.audit.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        for position, score_key in enumerate(sorted(scores), 1):
            messages = list(read_jsonl(temp/f'{score_key}.messages.jsonl'))
            sample_ids = {row["sample_id"] for row in messages}
            source_path = temp/f'{score_key}.source.jsonl'
            if not source_path.exists():
                raise RuntimeError(f'Missing full-score source for {score_key}')
            score_sources = list(read_jsonl(source_path))
            input_path = temp/f'{score_key}.messages.jsonl'
            audit_path = temp / f"{score_key}.audit.jsonl"
            if not audit_path.exists():
                raise RuntimeError(f'Missing audit for score {score_key}')
            output_name = f"{score_key}.html"
            command = [
                sys.executable, str(viewer),
                "--input", str(input_path), "--audit", str(audit_path),
                "--source", str(source_path), "--output", str(args.output_dir / output_name),
            ]
            if args.token_lengths:
                command.extend(['--token-lengths', str(args.token_lengths)])
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            phrase_count = len({(row.get("provenance") or {}).get("source_trajectory_id") for row in messages})
            title = next((str(row.get("title") or (row.get('input') or {}).get('metadata', {}).get('score_title') or '') for row in score_sources), '')
            entries.append((score_key, title, phrase_count, output_name))
            if args.notice:
                target=args.output_dir/output_name
                content=target.read_text(encoding='utf-8')
                content=content.replace('<body>', '<body><p style="padding:16px;color:#ffcc80">'+html.escape(args.notice)+'</p>', 1)
                target.write_text(content, encoding='utf-8')
            print(f"[{position}/{len(scores)}] {score_key}: {phrase_count}", flush=True)

    links = "\n".join(
        f'<li><a href="{html.escape(filename)}"><code>{html.escape(score_key)}</code>'
        f'{"｜" + html.escape(title) if title else ""}</a><span>{count} phrases</span></li>'
        for score_key, title, count, filename in entries
    )
    index = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>古琴 Agent 全量轨迹索引</title>
<style>body{{margin:0;background:#101216;color:#e8ecf1;font:15px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:920px;margin:auto;padding:28px}}h1{{font-size:24px}}p{{color:#a6b0bd}}ul{{list-style:none;padding:0;background:#181c22;border:1px solid #303741;border-radius:10px;overflow:hidden}}li{{display:flex;justify-content:space-between;gap:16px;padding:10px 14px;border-bottom:1px solid #303741}}li:last-child{{border:0}}a{{color:#8bb7ff;text-decoration:none}}span{{color:#a6b0bd;white-space:nowrap}}</style></head><body><main><h1>古琴 Agent 全量轨迹</h1>
<p>{len(entries)} 份曲谱；按曲目打开，避免一次加载全量轨迹。</p><p>{html.escape(args.notice)}</p><ul>{links}</ul></main></body></html>'''
    if args.token_lengths:
        with args.token_lengths.open(encoding='utf-8') as handle:
            token_rows = [r for r in csv.DictReader(handle, delimiter='\t')
                          if r['source_sample_id'] in sample_scores]
        token_rows.sort(key=lambda r: int(r['tokens']), reverse=True)
        ranked=[]
        for row in token_rows:
            sid=row['source_sample_id'];score=sample_scores[sid]
            phrase=sid.removesuffix('-fingering_agent-teacher-tools').removesuffix('-guqinization-teacher-tools')
            stage='Fingering' if row['agent_stage']=='fingering_agent' else 'Guqinizer'
            ranked.append(f'<li><a href="{html.escape(score)}.html?phrase={html.escape(phrase)}">'
                          f'{html.escape(phrase)} · {stage}</a><span>{int(row["tokens"]):,} tokens</span></li>')
        index=index.replace('<ul>'+links+'</ul>',
                            f'<h2>超长阶段（{len(ranked)}条，按token降序）</h2><ul>'+''.join(ranked)+
                            '</ul><h2>按曲目查看</h2><ul>'+links+'</ul>')
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir / "index.html"), "scores": len(entries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
