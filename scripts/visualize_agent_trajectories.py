#!/usr/bin/env python3
"""Render agent-messages JSONL as a readable, self-contained HTML report."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.qwen35_serialization import serialize_qwen35
from agents.abc_to_jianzipu.jianzi_renderer import render_jianzi_surface


def pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def block(title: str, body: str, *, opened: bool = False, css: str = "") -> str:
    return (
        f'<details class="block {css}" {"open" if opened else ""}>'
        f"<summary>{esc(title)}</summary><pre>{esc(body)}</pre></details>"
    )


def render_message(message: dict, number: int) -> str:
    role = message.get("role", "unknown")
    parts = [f'<section class="message {esc(role)}">',
             f'<h3><span class="num">{number}</span>{esc(role)}</h3>']
    content = message.get("content")
    if content:
        parts.append(block("内容", pretty(content), opened=role in {"system", "assistant"}))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name", "unknown")
        parts.append(block(f"调用工具 · {name}", pretty(function.get("arguments", {})),
                           opened=True, css="call"))
    if role == "tool":
        parts.append(f'<div class="meta">对应调用：{esc(message.get("tool_call_id", ""))}</div>')
    parts.append("</section>")
    return "".join(parts)


def parse_annotation_gqs(text: str) -> dict[int, dict[str, str]]:
    rows = {}
    for line in (text or "").splitlines():
        if not line.startswith("音｜"):
            continue
        cells = json.loads(line.split("｜", 1)[1])
        if len(cells) not in {3, 4}:
            continue
        rows[int(cells[0])] = {"jianpu": str(cells[1] or ""),
                               "jianzi": str(cells[2] or "")}
    return rows


def render_final_annotation_comparison(private: dict | None) -> str:
    payload = (private or {}).get("teacher_private") or {}
    annotation = parse_annotation_gqs(payload.get("annotation_gqs") or "")
    actions = ((payload.get("accepted_plan") or {}).get("actions") or [])
    if not annotation or not actions:
        return ""
    final = {int(action["source_index"]): action for action in actions}
    rows = []
    for index in sorted(set(annotation) | set(final)):
        annotated = annotation.get(index, {})
        annotation_text = annotated.get("jianzi", "")
        final_text = (render_jianzi_surface(final[index]).strip("[]")
                      if index in final else "")
        if not annotation_text and not final_text:
            continue
        same = annotation_text.strip() == final_text.strip()
        status = "一致" if same else "不同"
        css = "same" if same else "different"
        rows.append(
            f'<tr class="{css}"><td>{index}</td><td>{esc(annotated.get("jianpu", ""))}</td>'
            f'<td>{esc("[" + final_text + "]" if final_text else "—")}</td>'
            f'<td>{esc("[" + annotation_text + "]" if annotation_text else "—")}</td>'
            f'<td>{status}</td></tr>'
        )
    if not rows:
        return ""
    return (
        '<section class="comparison"><h3>最终版本 vs 标注版本</h3>'
        '<p>逐序号比较最终确定性重放结果与原始标注文字；“不同”不必然代表错误，'
        '也可能是合理替代指法或标注无法直接监督。</p>'
        '<div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th>'
        '<th>最终版本</th><th>标注版本</th><th>结果</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div></section>'
    )


def _plan_surface(action: dict | None) -> str:
    if action is None:
        return "—"
    explicit = action.get("jianzi_text")
    if explicit == "":
        return "[空]"
    if explicit is not None:
        return f"[{explicit}]"
    rendered = render_jianzi_surface(action).strip()
    return rendered if rendered else "—"


def _comparison_key(surface: str) -> str:
    """Normalize harmless display variants before coloring a comparison cell."""
    text = (surface or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return re.sub(r"\s+", "", text).translate(str.maketrans("一二三四五六七", "1234567"))


def render_cross_stage_comparison(samples: list[dict], private_by_id: dict[str, dict]) -> str:
    """Render one top-level Fingering/Guqinizer/annotation table per phrase."""
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        private = private_by_id.get(sample_id)
        if not private:
            continue
        provenance = sample.get("provenance") or {}
        key = str(provenance.get("source_trajectory_id") or sample_id)
        group = groups.setdefault(key, {"stages": {}, "annotation": {}})
        payload = private.get("teacher_private") or {}
        group["stages"][sample.get("agent_stage")] = (
            (payload.get("accepted_plan") or {}).get("actions") or []
        )
        annotation = parse_annotation_gqs(payload.get("annotation_gqs") or "")
        if annotation:
            group["annotation"] = annotation
    sections = []
    for trajectory_id, group in groups.items():
        stage_actions = {
            stage: {int(action["source_index"]): action for action in actions}
            for stage, actions in group["stages"].items()
        }
        annotation = group["annotation"]
        indices = set(annotation)
        for actions in stage_actions.values():
            indices.update(actions)
        rows = []
        for index in sorted(indices):
            annotated = annotation.get(index, {})
            finger = _plan_surface(stage_actions.get("fingering_agent", {}).get(index))
            guqin = _plan_surface(stage_actions.get("guqinization", {}).get(index))
            annotation_surface = (f'[{annotated["jianzi"]}]'
                                  if annotated.get("jianzi") else "[空]")
            finger_css = "match" if _comparison_key(finger) == _comparison_key(annotation_surface) else ""
            guqin_css = "match" if _comparison_key(guqin) == _comparison_key(annotation_surface) else ""
            rows.append(
                f'<tr><td>{index}</td><td>{esc(annotated.get("jianpu", ""))}</td>'
                f'<td class="{finger_css}">{esc(finger)}</td>'
                f'<td class="{guqin_css}">{esc(guqin)}</td>'
                f'<td>{esc(annotation_surface)}</td></tr>'
            )
        if rows:
            sections.append(
                f'<section class="stage-comparison"><h2>两阶段最终版本 vs 标注版本｜{esc(trajectory_id)}</h2>'
                '<div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th>'
                '<th>Fingering 最终版</th><th>Guqinizer 最终版</th><th>标注版本</th>'
                '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>'
            )
    return "".join(sections)


def render_sample(sample: dict, *, show_tools: bool = False,
                  serialized_prompt: str | None = None,
                  private: dict | None = None) -> str:
    tools = sample.get("tools") or []
    tool_names = " · ".join(str(tool.get("name")) for tool in tools)
    tool_badge = f'<span>{len(tools)} 个工具</span>' if show_tools else ""
    head = (
        f'<article class="sample"><header><h2>{esc(sample.get("sample_id"))}</h2>'
        f'<div class="badges"><span>{esc(sample.get("agent_stage"))}</span>'
        f'<span>{len(sample.get("messages") or [])} 条消息</span>'
        f'{tool_badge}</div></header>'
    )
    schemas = ""
    if show_tools:
        schemas = f'<details class="tools"><summary>可用工具：{esc(tool_names)}</summary>'
        schemas += "".join(block(tool.get("name", "unknown"), pretty(tool), css="schema")
                           for tool in tools)
        schemas += "</details>"
    serialized = ""
    if serialized_prompt is not None:
        serialized = block(
            "Qwen3.5-9B 官方 chat template 序列化结果",
            serialized_prompt,
            opened=True,
            css="serialized",
        )
    messages = "".join(render_message(message, index)
                       for index, message in enumerate(sample.get("messages") or [], 1))
    return head + schemas + serialized + messages + "</article>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="messages_train.jsonl")
    parser.add_argument("-o", "--output", type=Path, help="output HTML path")
    parser.add_argument("--sample-id", action="append", help="render only selected sample IDs")
    parser.add_argument("--companion-input", type=Path, action="append", default=[],
                        help="optional prior-stage messages JSONL to combine into the comparison")
    parser.add_argument("--companion-stage", action="append", default=[],
                        help="only include these agent stages from companion inputs")
    parser.add_argument("--show-tools", action="store_true",
                        help="include tool schemas; by default the report shows messages only")
    parser.add_argument("--qwen35-template", action="store_true",
                        help="render the official Qwen3.5-9B serialized training prompt")
    parser.add_argument("--audit", type=Path,
                        help="private audit JSONL; defaults to the input directory")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".html")
    wanted = set(args.sample_id or [])
    samples = []
    input_paths = [args.input, *args.companion_input]
    companion_stages = set(args.companion_stage)
    for position, input_path in enumerate(input_paths):
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if (wanted and item.get("sample_id") not in wanted
                        or position and companion_stages
                        and item.get("agent_stage") not in companion_stages):
                    continue
                samples.append(item)
    audit_path = args.audit or args.input.with_name("teacher_trajectory_audit.jsonl")
    private_by_id = {}
    audit_paths = [audit_path]
    if args.audit is None:
        audit_paths.extend(
            input_path.with_name("teacher_trajectory_audit.jsonl")
            for input_path in args.companion_input
        )
    for candidate_audit in audit_paths:
        if candidate_audit.exists():
            with candidate_audit.open(encoding="utf-8") as handle:
                private_by_id.update({
                    item.get("sample_id"): item
                    for line in handle if (item := json.loads(line))
                })
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent 训练轨迹</title><style>
:root{{--bg:#f5f3ee;--paper:#fffdfa;--ink:#25231f;--muted:#746f65;--line:#ddd7ca;
--system:#65558f;--user:#176b87;--assistant:#39734e;--tool:#a65d19}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1100px;margin:auto;padding:32px 20px}}h1{{font-size:28px;margin:0 0 6px}}.intro{{color:var(--muted);margin-bottom:28px}}
.sample{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:0 0 30px;overflow:hidden;box-shadow:0 6px 24px #594d3510}}
.sample>header{{padding:20px 24px;border-bottom:1px solid var(--line)}}h2{{font-size:19px;margin:0 0 9px;overflow-wrap:anywhere}}
.badges span{{display:inline-block;background:#ede9df;border-radius:99px;padding:2px 9px;margin:0 6px 4px 0;font-size:12px}}
details.tools{{padding:13px 24px;border-bottom:1px solid var(--line)}}summary{{cursor:pointer;font-weight:650}}
.message{{margin:18px 24px;padding:14px 16px;border-left:5px solid;border-radius:7px;background:#fff}}
.message.system{{border-color:var(--system)}}.message.user{{border-color:var(--user)}}.message.assistant{{border-color:var(--assistant)}}.message.tool{{border-color:var(--tool)}}
h3{{font-size:15px;margin:0 0 8px;text-transform:capitalize}}.num{{display:inline-grid;place-items:center;width:24px;height:24px;margin-right:8px;border-radius:50%;background:#ece8df;font-size:12px}}
.block{{margin:8px 0;border:1px solid var(--line);border-radius:6px;overflow:hidden}}.block summary{{padding:7px 10px;background:#f7f4ed;font-size:13px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#fcfbf7;font:13px/1.55 ui-monospace,"Cascadia Code",monospace}}
.call{{border-color:#d5b28c}}.call summary{{background:#f9ead8}}.schema{{margin-left:12px}}.meta{{font-size:12px;color:var(--muted);margin-top:6px}}
.comparison{{margin:18px 24px;padding:14px 16px;border:1px solid var(--line);border-radius:8px;background:#fff}}
.comparison h3{{font-size:17px;margin:0 0 4px}}.comparison p{{color:var(--muted);font-size:13px;margin:0 0 10px}}
.stage-comparison{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:0 0 30px;padding:20px 24px;overflow:hidden;box-shadow:0 6px 24px #594d3510}}.stage-comparison h2{{font-size:19px;margin:0 0 14px;overflow-wrap:anywhere}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px 9px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#f2eee5}}td.match{{background:#e7f5e9;color:#21683a;font-weight:700}}tr.same td:last-child{{color:#347348;font-weight:700}}tr.different{{background:#fff4e8}}tr.different td:last-child{{color:#a65316;font-weight:700}}
</style></head><body><main><h1>Agent 训练轨迹</h1>
{render_cross_stage_comparison(samples, private_by_id)}
<div class="intro">来源：{esc(args.input)} · 共 {len(samples)} 条。按时间顺序显示 messages；点击标题可展开正文和调用参数。若存在私有审计文件，页面同时显示最终版本与标注版本对比；对比内容不得进入训练。</div>
{"".join(render_sample(
    sample,
    show_tools=args.show_tools,
    serialized_prompt=serialize_qwen35(sample) if args.qwen35_template else None,
    private=private_by_id.get(sample.get("sample_id")),
) for sample in samples)}</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"input": str(args.input), "output": str(output),
                      "samples": len(samples)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
