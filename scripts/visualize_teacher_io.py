#!/usr/bin/env python3
"""Render private teacher request/response traces without altering provider structure."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pretty(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def details(title: str, value, *, opened: bool = False, css: str = "") -> str:
    return (f'<details class="{css}" {"open" if opened else ""}>'
            f'<summary>{esc(title)}</summary><pre>{esc(pretty(value))}</pre></details>')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="teacher_trajectory_audit.jsonl")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name("teacher_io_viewer.html")
    samples = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    articles = []
    nav = []
    for sample_number, sample in enumerate(samples, 1):
        rounds = sample.get("teacher_private", {}).get("teacher_io_trace") or []
        stage = sample.get("agent_stage") or "unknown"
        anchor = f"sample-{sample_number}"
        nav.append(f'<a href="#{anchor}">{esc(stage)} · {esc(sample.get("sample_id"))}</a>')
        rendered_rounds = []
        for item in rounds:
            request = item.get("request") or {}
            request_without_tools = {key: value for key, value in request.items()
                                     if key not in {"tools", "tool_choice"}}
            rendered_rounds.append(
                f'<section><h3>第 {item.get("round")} 轮</h3>'
                + details("真实请求：system + messages", request_without_tools, opened=True)
                + details("真实请求：tools", request.get("tools") or [], css="tools")
                + details("真实请求：tool_choice", request.get("tool_choice"))
                + details("教师原始响应", item.get("response"), opened=True, css="response")
                + "</section>"
            )
        articles.append(
            f'<article id="{anchor}"><h2>{esc(stage)}｜{esc(sample.get("sample_id"))}</h2>'
            f'<p>{len(rounds)} 轮真实请求／响应；此页含私有教师信息，不得用于训练。</p>'
            + "".join(rendered_rounds) + "</article>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>教师真实 I/O</title>
<style>body{{max-width:1100px;margin:30px auto;padding:0 20px;background:#f5f2eb;color:#28251f;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
article{{background:#fffdf8;border:1px solid #d9d2c4;border-radius:12px;padding:20px;margin:24px 0}}nav{{position:sticky;top:0;background:#f5f2eb;padding:8px 0;display:flex;gap:10px;flex-wrap:wrap}}nav a{{background:#e8f1f4;border:1px solid #c6d7db;border-radius:999px;padding:4px 10px;color:#24505a;text-decoration:none;font-size:13px}}section{{border-top:1px solid #ddd5c7;margin-top:18px;padding-top:12px}}
details{{border:1px solid #d9d2c4;border-radius:6px;margin:8px 0;overflow:hidden}}summary{{cursor:pointer;font-weight:650;padding:8px 10px;background:#f2eee5}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#fff;font:12px/1.5 ui-monospace,"Cascadia Code",monospace}}
.tools summary{{background:#e8f1f4}}.response summary{{background:#e9f2e7}}p{{color:#756d61}}</style></head>
<body><h1>教师模型真实输入／输出</h1><p>保持提供商原生请求结构；tools 独立保存，不拼入 messages。</p>
<nav>{"".join(nav)}</nav>
{"".join(articles)}</body></html>"""
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"input": str(args.input), "output": str(output),
                      "samples": len(samples)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
