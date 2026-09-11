#!/usr/bin/env python3
"""Render many agent-training trajectories from one or more JSONL files as a self-contained HTML report.

Compared with the original single/flat viewer, this version groups samples by
``provenance.source_trajectory_id`` and provides client-side trajectory
navigation, search, stage filtering, expand/collapse controls, and summary
statistics while preserving the original message/tool/audit comparison views.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable


# Keep the original project-layout behavior, but do not require project-only
# modules for the basic HTML viewer.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_project_helpers() -> tuple[Callable[[dict], str] | None, Callable[[dict], str] | None]:
    try:
        from agents.abc_to_jianzipu.qwen35_serialization import serialize_qwen35
        from agents.abc_to_jianzipu.jianzi_renderer import render_jianzi_surface
        return serialize_qwen35, render_jianzi_surface
    except (ImportError, ModuleNotFoundError):
        return None, None


SERIALIZE_QWEN35, RENDER_JIANZI_SURFACE = _load_project_helpers()


def pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def attr(value: Any) -> str:
    return esc(value)


def natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value or "")
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def block(title: str, body: str, *, opened: bool = False, css: str = "") -> str:
    return (
        f'<details class="block {css}" {"open" if opened else ""}>'
        f"<summary>{esc(title)}</summary><pre>{esc(body)}</pre></details>"
    )


def render_message(message: dict, number: int) -> str:
    role = str(message.get("role", "unknown"))
    parts = [
        f'<section class="message {attr(role)}">',
        f'<h4><span class="num">{number}</span>{esc(role)}</h4>',
    ]
    content = message.get("content")
    if content not in {None, ""}:
        parts.append(block("内容", pretty(content), opened=role in {"system", "assistant"}))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name", "unknown")
        parts.append(
            block(
                f"调用工具 · {name}",
                pretty(function.get("arguments", {})),
                opened=True,
                css="call",
            )
        )
    if role == "tool":
        tool_name = message.get("name") or ""
        meta_bits = []
        if tool_name:
            meta_bits.append(f"工具：{esc(tool_name)}")
        if message.get("tool_call_id"):
            meta_bits.append(f"对应调用：{esc(message.get('tool_call_id'))}")
        if meta_bits:
            parts.append(f'<div class="meta">{" · ".join(meta_bits)}</div>')
    parts.append("</section>")
    return "".join(parts)


def parse_annotation_gqs(text: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for line in (text or "").splitlines():
        if not line.startswith("音｜"):
            continue
        try:
            cells = json.loads(line.split("｜", 1)[1])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(cells, list) or len(cells) not in {3, 4}:
            continue
        try:
            index = int(cells[0])
        except (TypeError, ValueError):
            continue
        rows[index] = {
            "jianpu": str(cells[1] or ""),
            "jianzi": str(cells[2] or ""),
        }
    return rows


def _fallback_action_surface(action: dict | None) -> str:
    if action is None:
        return "—"
    explicit = action.get("jianzi_text")
    if explicit == "":
        return "[空]"
    if explicit is not None:
        text = str(explicit).strip()
        return text if text.startswith("[") else f"[{text}]"
    # Common fallback for audit formats that keep rows instead of renderer fields.
    for key in ("jianzi", "surface", "text"):
        value = action.get(key)
        if value is not None:
            text = str(value).strip()
            if not text:
                return "[空]"
            return text if text.startswith("[") else f"[{text}]"
    return "—"


def _plan_surface(action: dict | None) -> str:
    if action is None:
        return "—"
    explicit = action.get("jianzi_text")
    if explicit == "":
        return "[空]"
    if explicit is not None:
        text = str(explicit).strip()
        return text if text.startswith("[") else f"[{text}]"
    if RENDER_JIANZI_SURFACE is not None:
        try:
            rendered = RENDER_JIANZI_SURFACE(action).strip()
            return rendered if rendered else "—"
        except Exception:
            pass
    return _fallback_action_surface(action)


def _comparison_key(surface: str) -> str:
    """Normalize harmless display variants before coloring a comparison cell."""
    text = (surface or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return re.sub(r"\s+", "", text).translate(str.maketrans("一二三四五六七", "1234567"))


def _private_payload(private: dict | None) -> dict:
    return ((private or {}).get("teacher_private") or {})


def build_cross_stage_group(samples: list[dict], private_by_id: dict[str, dict]) -> dict[str, Any] | None:
    stages: dict[str, list[dict]] = {}
    annotation: dict[int, dict[str, str]] = {}
    for sample in samples:
        private = private_by_id.get(str(sample.get("sample_id") or ""))
        if not private:
            continue
        payload = _private_payload(private)
        actions = ((payload.get("accepted_plan") or {}).get("actions") or [])
        stages[str(sample.get("agent_stage") or "unknown")] = actions
        parsed = parse_annotation_gqs(payload.get("annotation_gqs") or "")
        if parsed:
            annotation = parsed
    if not stages and not annotation:
        return None
    return {"stages": stages, "annotation": annotation}


def render_cross_stage_comparison(trajectory_id: str, samples: list[dict], private_by_id: dict[str, dict]) -> str:
    group = build_cross_stage_group(samples, private_by_id)
    if not group:
        return ""

    stage_actions: dict[str, dict[int, dict]] = {}
    for stage, actions in group["stages"].items():
        by_index: dict[int, dict] = {}
        for action in actions:
            try:
                by_index[int(action["source_index"])] = action
            except (KeyError, TypeError, ValueError):
                continue
        stage_actions[stage] = by_index

    annotation = group["annotation"]
    indices = set(annotation)
    for actions in stage_actions.values():
        indices.update(actions)
    if not indices:
        return ""

    has_guqinizer = "guqinization" in stage_actions

    rows = []
    for index in sorted(indices):
        annotated = annotation.get(index, {})
        finger = _plan_surface(stage_actions.get("fingering_agent", {}).get(index))
        guqin = _plan_surface(stage_actions.get("guqinization", {}).get(index))
        annotation_surface = f'[{annotated["jianzi"]}]' if annotated.get("jianzi") else "[空]"
        finger_css = "match" if _comparison_key(finger) == _comparison_key(annotation_surface) else ""
        guqin_css = "match" if _comparison_key(guqin) == _comparison_key(annotation_surface) else ""
        guqin_cell = (
            f'<td class="{guqin_css}">{esc(guqin)}</td>'
            if has_guqinizer else ""
        )
        rows.append(
            f'<tr><td>{index}</td><td>{esc(annotated.get("jianpu", ""))}</td>'
            f'<td class="{finger_css}">{esc(finger)}</td>'
            f'{guqin_cell}<td>{esc(annotation_surface)}</td></tr>'
        )

    stage_heading = (
        "两阶段最终版本 vs 标注版本"
        if has_guqinizer else "Fingering 最终版本 vs 标注版本"
    )
    guqin_heading = "<th>Guqinizer 最终版</th>" if has_guqinizer else ""
    note = (
        ""
        if has_guqinizer else
        '<p class="comparison-note">本轨迹未生成 Guqinizer；这只表示结构化审计未发现可提交的后续 patch，不表示 Fingering 文本已与标注逐字一致。</p>'
    )

    return (
        '<section class="stage-comparison">'
        f'<div class="section-title"><h3>{stage_heading}</h3><span>{esc(trajectory_id)}</span></div>'
        f'{note}'
        '<div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th>'
        f'<th>Fingering 最终版</th>{guqin_heading}<th>标注版本</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def bool_badge(label: str, value: Any) -> str:
    if value is True:
        return f'<span class="badge ok">✓ {esc(label)}</span>'
    if value is False:
        return f'<span class="badge bad">✕ {esc(label)}</span>'
    return ""


def render_verification(sample: dict) -> str:
    verification = sample.get("verification") or {}
    if not isinstance(verification, dict) or not verification:
        return ""
    labels = {
        "tools_really_executed": "工具已执行",
        "replay_passed": "Replay",
        "target_fields_passed": "目标字段",
        "private_leakage_passed": "无私有泄漏",
        "training_eligible": "可训练",
    }
    badges = [bool_badge(labels.get(key, key), value) for key, value in verification.items()]
    badges = [item for item in badges if item]
    return f'<div class="verification">{"".join(badges)}</div>' if badges else ""


def serialize_if_requested(sample: dict, enabled: bool) -> str | None:
    if not enabled:
        return None
    if SERIALIZE_QWEN35 is None:
        raise RuntimeError(
            "--qwen35-template 需要项目模块 agents.abc_to_jianzipu.qwen35_serialization；"
            "当前环境未找到该模块。请在原项目目录中运行此脚本。"
        )
    return SERIALIZE_QWEN35(sample)


def render_sample(
    sample: dict,
    *,
    show_tools: bool = False,
    serialized_prompt: str | None = None,
    private: dict | None = None,
    opened: bool = False,
) -> str:
    tools = sample.get("tools") or []
    tool_names = " · ".join(str(tool.get("name")) for tool in tools)
    tool_badge = f'<span class="badge neutral">{len(tools)} 个工具</span>' if show_tools else ""
    token_count = sample.get("token_count")
    token_badge = (
        f'<span class="badge neutral">{esc(token_count)} tokens</span>'
        if isinstance(token_count, int) else ""
    )
    stage = str(sample.get("agent_stage") or "unknown")
    sample_id = str(sample.get("sample_id") or "")
    termination = (sample.get("termination") or {}).get("kind") if isinstance(sample.get("termination"), dict) else None
    termination_badge = f'<span class="badge neutral">{esc(termination)}</span>' if termination else ""
    source_file = str(sample.get("_source_file") or "")

    head = (
        f'<details class="sample" data-stage="{attr(stage)}" data-sample="{attr(sample_id.lower())}" {"open" if opened else ""}>'
        f'<summary class="sample-head"><div class="sample-title"><strong>{esc(sample_id)}</strong>'
        f'<span class="source-file">{esc(source_file)}</span></div>'
        f'<div class="badges"><span class="badge stage">{esc(stage)}</span>'
        f'<span class="badge neutral">{len(sample.get("messages") or [])} 条消息</span>'
        f'{token_badge}{tool_badge}{termination_badge}</div></summary>'
    )

    schemas = ""
    if show_tools:
        schemas = f'<details class="tools"><summary>可用工具：{esc(tool_names)}</summary>'
        schemas += "".join(
            block(str(tool.get("name", "unknown")), pretty(tool), css="schema") for tool in tools
        )
        schemas += "</details>"

    serialized = ""
    if serialized_prompt is not None:
        serialized = block(
            "Qwen3.5-9B 官方 chat template 序列化结果",
            serialized_prompt,
            opened=True,
            css="serialized",
        )

    verification = render_verification(sample)
    messages = "".join(
        render_message(message, index)
        for index, message in enumerate(sample.get("messages") or [], 1)
    )
    return head + verification + schemas + serialized + messages + "</details>"


def trajectory_id_for(sample: dict) -> str:
    provenance = sample.get("provenance") or {}
    value = provenance.get("source_trajectory_id") if isinstance(provenance, dict) else None
    if value:
        return str(value)
    sample_id = str(sample.get("sample_id") or "")
    # Conservative fallback: strip common stage suffixes only when present.
    for marker in ("-fingering_agent-", "-guqinization-"):
        if marker in sample_id:
            return sample_id.split(marker, 1)[0]
    return sample_id or "unknown-trajectory"


def score_key_for(sample: dict) -> str:
    """Return the score-level key used by the two-level navigator."""
    provenance = sample.get("provenance") or {}
    value = provenance.get("score_key") if isinstance(provenance, dict) else None
    if value:
        return str(value)
    trajectory_id = trajectory_id_for(sample)
    match = re.match(r"(.+?)-p\d+$", trajectory_id)
    return match.group(1) if match else trajectory_id


def is_noop_trajectory(samples: list[dict], private_by_id: dict[str, dict]) -> bool:
    """A no-op label is based on the Guqinizer audit, not stage absence."""
    guqinizer = next(
        (sample for sample in samples if str(sample.get("agent_stage") or "") == "guqinization"),
        None,
    )
    if guqinizer is None:
        return False
    private = private_by_id.get(str(guqinizer.get("sample_id") or ""))
    if not private:
        return False
    target_patches = (_private_payload(private).get("target_patches") or [])
    return isinstance(target_patches, list) and len(target_patches) == 0


def load_jsonl(paths: Iterable[Path], wanted: set[str], companion_stages: set[str], primary_count: int) -> list[dict]:
    samples: list[dict] = []
    for position, input_path in enumerate(paths):
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        with input_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{input_path}:{lineno}: JSON 解析失败: {exc}") from exc
                if not isinstance(item, dict):
                    continue
                if wanted and item.get("sample_id") not in wanted:
                    continue
                is_companion = position >= primary_count
                if is_companion and companion_stages and item.get("agent_stage") not in companion_stages:
                    continue
                item = dict(item)
                item["_source_file"] = input_path.name
                samples.append(item)
    return samples


def load_audits(paths: Iterable[Path]) -> dict[str, dict]:
    private_by_id: dict[str, dict] = {}
    seen: set[Path] = set()
    for candidate in paths:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        with candidate.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{candidate}:{lineno}: JSON 解析失败: {exc}") from exc
                if isinstance(item, dict) and item.get("sample_id"):
                    private_by_id[str(item["sample_id"])] = item
    return private_by_id


def group_trajectories(samples: list[dict]) -> OrderedDict[str, list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for sample in samples:
        groups.setdefault(trajectory_id_for(sample), []).append(sample)
    return groups


def render_sidebar_item(
    trajectory_id: str,
    samples: list[dict],
    index: int,
    private_by_id: dict[str, dict],
) -> str:
    stages = Counter(str(sample.get("agent_stage") or "unknown") for sample in samples)
    stage_text = " · ".join(f"{stage}×{count}" for stage, count in sorted(stages.items()))
    noop = is_noop_trajectory(samples, private_by_id)
    noop_badge = '<span class="badge noop">no-op</span>' if noop else ""
    searchable = " ".join(
        [trajectory_id]
        + [str(sample.get("sample_id") or "") for sample in samples]
        + list(stages)
    ).lower()
    return (
        f'<button type="button" class="trajectory-link" data-target="traj-{index}" '
        f'data-search="{attr(searchable)}" data-stages="{attr(" ".join(stages))}">'
        f'<span class="trajectory-name">{esc(trajectory_id)} {noop_badge}</span>'
        f'<span class="trajectory-meta">{len(samples)} sample · {esc(stage_text)}</span>'
        "</button>"
    )


def render_score_group(
    score_key: str,
    entries: list[tuple[str, list[dict], int]],
    private_by_id: dict[str, dict],
) -> str:
    """Render the score-level node containing phrase-level trajectory links."""
    links = "".join(
        render_sidebar_item(trajectory_id, samples, index, private_by_id)
        for trajectory_id, samples, index in entries
    )
    noop_count = sum(1 for _, samples, _ in entries if is_noop_trajectory(samples, private_by_id))
    noop_text = f" · no-op {noop_count}" if noop_count else ""
    searchable = " ".join(
        [score_key]
        + [trajectory_id for trajectory_id, _, _ in entries]
        + [str(sample.get("sample_id") or "") for _, samples, _ in entries for sample in samples]
    ).lower()
    return (
        f'<details class="score-group" data-search="{attr(searchable)}" open>'
        f'<summary><span class="score-name">{esc(score_key)}</span>'
        f'<span class="score-meta">{len(entries)} phrases{esc(noop_text)}</span></summary>'
        f'<div class="score-phrases">{links}</div></details>'
    )


def render_trajectory_panel(
    trajectory_id: str,
    samples: list[dict],
    index: int,
    private_by_id: dict[str, dict],
    *,
    show_tools: bool,
    qwen35_template: bool,
) -> str:
    stage_counts = Counter(str(sample.get("agent_stage") or "unknown") for sample in samples)
    message_count = sum(len(sample.get("messages") or []) for sample in samples)
    training_eligible = sum(
        1
        for sample in samples
        if isinstance(sample.get("verification"), dict)
        and sample["verification"].get("training_eligible") is True
    )
    stage_badges = "".join(
        f'<span class="badge stage">{esc(stage)} × {count}</span>'
        for stage, count in sorted(stage_counts.items())
    )
    comparison = render_cross_stage_comparison(trajectory_id, samples, private_by_id)
    sample_html = "".join(
        render_sample(
            sample,
            show_tools=show_tools,
            serialized_prompt=serialize_if_requested(sample, qwen35_template),
            private=private_by_id.get(str(sample.get("sample_id") or "")),
            opened=(sample_index == 0),
        )
        for sample_index, sample in enumerate(samples)
    )
    searchable = " ".join(
        [trajectory_id]
        + [str(sample.get("sample_id") or "") for sample in samples]
        + list(stage_counts)
    ).lower()
    return f"""
<section class="trajectory-panel" id="traj-{index}" data-trajectory="{attr(trajectory_id)}" data-search="{attr(searchable)}">
  <div class="trajectory-header">
    <div>
      <div class="eyebrow">Trajectory {index + 1}</div>
      <h2>{esc(trajectory_id)}</h2>
      <div class="badges">{stage_badges}<span class="badge neutral">{len(samples)} sample</span><span class="badge neutral">{message_count} 条消息</span><span class="badge ok">{training_eligible}/{len(samples)} 可训练</span></div>
    </div>
    <div class="trajectory-actions">
      <button type="button" class="small-btn prev-btn">← 上一个</button>
      <button type="button" class="small-btn next-btn">下一个 →</button>
      <button type="button" class="small-btn expand-btn">展开全部</button>
      <button type="button" class="small-btn collapse-btn">收起全部</button>
    </div>
  </div>
  {comparison}
  <div class="samples">{sample_html}</div>
</section>
"""


def build_page(
    *,
    input_paths: list[Path],
    samples: list[dict],
    private_by_id: dict[str, dict],
    show_tools: bool,
    qwen35_template: bool,
) -> str:
    groups = group_trajectories(samples)
    stage_counts = Counter(str(sample.get("agent_stage") or "unknown") for sample in samples)
    message_total = sum(len(sample.get("messages") or []) for sample in samples)
    tool_call_total = sum(
        len(message.get("tool_calls") or [])
        for sample in samples
        for message in sample.get("messages") or []
    )
    eligible_total = sum(
        1
        for sample in samples
        if isinstance(sample.get("verification"), dict)
        and sample["verification"].get("training_eligible") is True
    )

    stage_options = "".join(
        f'<option value="{attr(stage)}">{esc(stage)} ({count})</option>'
        for stage, count in sorted(stage_counts.items())
    )
    score_groups: OrderedDict[str, list[tuple[str, list[dict], int]]] = OrderedDict()
    for index, (trajectory_id, trajectory_samples) in enumerate(groups.items()):
        score_groups.setdefault(score_key_for(trajectory_samples[0]), []).append(
            (trajectory_id, trajectory_samples, index)
        )
    sidebar_items = "".join(
        render_score_group(score_key, entries, private_by_id)
        for score_key, entries in score_groups.items()
    )
    panels = "".join(
        render_trajectory_panel(
            trajectory_id,
            trajectory_samples,
            index,
            private_by_id,
            show_tools=show_tools,
            qwen35_template=qwen35_template,
        )
        for index, (trajectory_id, trajectory_samples) in enumerate(groups.items())
    )
    sources = "、".join(path.name for path in input_paths)
    ranked = bool(samples) and all(isinstance(sample.get("token_count"), int) for sample in samples)
    page_title = "最长轨迹 Top 10" if ranked else "Agent 多轨迹可视化"
    intro = (
        "按 Qwen3.5 chat-template 实际 token 数从长到短排列；展开样本可查看完整多轮 messages、工具调用和最终回复。"
        if ranked else
        "按 trajectory 分组；左侧搜索/筛选，右侧查看完整 messages、工具调用、可选 Qwen3.5 序列化及私有审计对比。"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(page_title)}</title>
<style>
:root{{--bg:#f5f3ee;--paper:#fffdfa;--ink:#25231f;--muted:#746f65;--line:#ddd7ca;--system:#65558f;--user:#176b87;--assistant:#39734e;--tool:#a65d19;--accent:#345f88;--good:#2f7045;--bad:#a44c35;--sidebar:#eeeae1}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
button,input,select{{font:inherit}}
.shell{{max-width:1500px;margin:auto;padding:24px}}
.page-head{{margin-bottom:18px}}
h1{{font-size:28px;margin:0 0 4px}}.intro{{color:var(--muted);margin:0}}
.stats{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:18px 0}}
.stat{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.stat strong{{display:block;font-size:22px;line-height:1.2}}.stat span{{color:var(--muted);font-size:12px}}
.workspace{{display:grid;grid-template-columns:310px minmax(0,1fr);gap:18px;align-items:start}}
.sidebar{{position:sticky;top:12px;background:var(--sidebar);border:1px solid var(--line);border-radius:12px;padding:12px;max-height:calc(100vh - 24px);overflow:auto}}
.filters{{display:grid;gap:8px;margin-bottom:10px}}
.search,.stage-filter{{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink)}}
.result-count{{font-size:12px;color:var(--muted);margin:3px 2px 8px}}
.trajectory-list{{display:grid;gap:8px}}.score-group{{border:1px solid var(--line);border-radius:8px;background:#f6f2e9;overflow:hidden}}.score-group>summary{{padding:8px 10px;display:flex;justify-content:space-between;gap:8px;align-items:baseline;cursor:pointer;list-style:none}}.score-group>summary::-webkit-details-marker{{display:none}}.score-group>summary::before{{content:"▸";color:var(--muted);margin-right:5px}}.score-group[open]>summary::before{{content:"▾"}}.score-name{{font-weight:700;overflow-wrap:anywhere}}.score-meta{{font-size:11px;color:var(--muted);white-space:nowrap}}.score-phrases{{display:grid;gap:4px;padding:0 6px 6px}}
.trajectory-link{{width:100%;text-align:left;border:1px solid transparent;background:transparent;border-radius:8px;padding:9px 10px;cursor:pointer;color:var(--ink)}}
.trajectory-link:hover{{background:#ffffff80;border-color:var(--line)}}.trajectory-link.active{{background:var(--paper);border-color:#b9c7d4;box-shadow:0 2px 10px #594d3512}}
.trajectory-name{{display:block;font-weight:700;overflow-wrap:anywhere}}.trajectory-meta{{display:block;color:var(--muted);font-size:11px;margin-top:2px;overflow-wrap:anywhere}}.badge.noop{{background:#f4ead1;color:#87621d}}
.content{{min-width:0}}
.trajectory-panel{{display:none}}.trajectory-panel.active{{display:block}}
.trajectory-header{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px;display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}
.trajectory-header h2{{font-size:22px;margin:1px 0 8px;overflow-wrap:anywhere}}.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}
.trajectory-actions{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}
.small-btn{{border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 9px;cursor:pointer;color:var(--ink)}}.small-btn:hover{{border-color:#a9a294;background:#faf7f0}}
.badges{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}.badge{{display:inline-block;border-radius:99px;padding:2px 8px;font-size:11px;line-height:1.7;background:#ede9df;color:#5b574f}}
.badge.stage{{background:#e8edf3;color:#365674}}.badge.ok{{background:#e5f2e8;color:var(--good)}}.badge.bad{{background:#f7e7e3;color:var(--bad)}}.badge.neutral{{background:#ede9df;color:#5b574f}}
.sample{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:0 0 14px;overflow:hidden;box-shadow:0 4px 18px #594d350d}}
.sample[open]{{border-color:#c8c0b1}}.sample-head{{display:flex;justify-content:space-between;gap:12px;padding:16px 18px;cursor:pointer;list-style:none}}.sample-head::-webkit-details-marker{{display:none}}
.sample-head::before{{content:"▸";margin-right:8px;color:var(--muted);align-self:flex-start}}.sample[open]>.sample-head::before{{content:"▾"}}
.sample-title{{min-width:0;flex:1}}.sample-title strong{{display:block;font-size:15px;overflow-wrap:anywhere}}.source-file{{display:block;font-size:11px;color:var(--muted);margin-top:2px}}
.verification{{padding:0 18px 12px;display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line)}}
details.tools{{padding:12px 18px;border-bottom:1px solid var(--line)}}summary{{cursor:pointer;font-weight:650}}
.message{{margin:16px 18px;padding:12px 14px;border-left:5px solid;border-radius:7px;background:#fff}}
.message.system{{border-color:var(--system)}}.message.user{{border-color:var(--user)}}.message.assistant{{border-color:var(--assistant)}}.message.tool{{border-color:var(--tool)}}
h3{{font-size:17px;margin:0}}h4{{font-size:14px;margin:0 0 8px;text-transform:none}}.num{{display:inline-grid;place-items:center;width:24px;height:24px;margin-right:8px;border-radius:50%;background:#ece8df;font-size:12px}}
.block{{margin:8px 0;border:1px solid var(--line);border-radius:6px;overflow:hidden}}.block summary{{padding:7px 10px;background:#f7f4ed;font-size:13px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#fcfbf7;font:13px/1.55 ui-monospace,"Cascadia Code",monospace}}
.call{{border-color:#d5b28c}}.call summary{{background:#f9ead8}}.schema{{margin-left:12px}}.meta{{font-size:12px;color:var(--muted);margin-top:6px}}
.comparison{{margin:12px 18px 16px;padding:12px 14px;border:1px solid var(--line);border-radius:8px;background:#fff}}.comparison h4{{font-size:15px;margin:0 0 4px}}.comparison p{{color:var(--muted);font-size:12px;margin:0 0 9px}}
.stage-comparison{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:0 0 14px;padding:16px 18px;overflow:hidden}}.comparison-note{{color:var(--muted);font-size:12px;margin:-3px 0 10px}}.section-title{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:10px}}.section-title span{{font-size:11px;color:var(--muted);overflow-wrap:anywhere}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:7px 9px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#f2eee5;position:sticky;top:0}}td.match{{background:#e7f5e9;color:#21683a;font-weight:700}}tr.same td:last-child{{color:#347348;font-weight:700}}tr.different{{background:#fff4e8}}tr.different td:last-child{{color:#a65316;font-weight:700}}
.empty{{display:none;background:var(--paper);border:1px dashed var(--line);border-radius:12px;padding:28px;text-align:center;color:var(--muted)}}
@media(max-width:900px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}.workspace{{grid-template-columns:1fr}}.sidebar{{position:static;max-height:none}}.trajectory-header{{display:block}}.trajectory-actions{{justify-content:flex-start;margin-top:12px}}}}
@media(max-width:560px){{.shell{{padding:14px}}.stats{{grid-template-columns:1fr 1fr}}.sample-head{{display:block}}.sample-head .badges{{margin-top:8px}}.section-title{{display:block}}}}
</style>
</head>
<body>
<div class="shell">
  <header class="page-head">
    <h1>{esc(page_title)}</h1>
    <p class="intro">来源：{esc(sources)}。{esc(intro)}</p>
  </header>

  <section class="stats">
    <div class="stat"><strong>{len(groups)}</strong><span>Trajectories</span></div>
    <div class="stat"><strong>{len(samples)}</strong><span>Samples</span></div>
    <div class="stat"><strong>{message_total}</strong><span>Messages</span></div>
    <div class="stat"><strong>{tool_call_total}</strong><span>Tool calls</span></div>
    <div class="stat"><strong>{eligible_total}/{len(samples)}</strong><span>Training eligible</span></div>
  </section>

  <div class="workspace">
    <aside class="sidebar">
      <div class="filters">
        <input id="search" class="search" type="search" placeholder="搜索 trajectory / sample..." autocomplete="off">
        <select id="stageFilter" class="stage-filter" aria-label="按 agent stage 筛选">
          <option value="">全部 stage</option>{stage_options}
        </select>
      </div>
      <div id="resultCount" class="result-count">{len(groups)} / {len(groups)} trajectories</div>
      <nav id="trajectoryList" class="trajectory-list">{sidebar_items}</nav>
    </aside>

    <main class="content">
      <div id="empty" class="empty">没有符合当前筛选条件的 trajectory。</div>
      {panels}
    </main>
  </div>
</div>
<script>
(() => {{
  const links = Array.from(document.querySelectorAll('.trajectory-link'));
  const scoreGroups = Array.from(document.querySelectorAll('.score-group'));
  const panels = Array.from(document.querySelectorAll('.trajectory-panel'));
  const search = document.getElementById('search');
  const stageFilter = document.getElementById('stageFilter');
  const resultCount = document.getElementById('resultCount');
  const empty = document.getElementById('empty');
  let visibleLinks = links.slice();
  let activeId = null;

  function panelFor(target) {{ return document.getElementById(target); }}

  function activate(target, updateHash = true) {{
    const link = links.find(x => x.dataset.target === target && !x.hidden);
    if (!link) return;
    activeId = target;
    links.forEach(x => x.classList.toggle('active', x === link));
    panels.forEach(p => p.classList.toggle('active', p.id === target));
    if (updateHash) history.replaceState(null, '', '#' + target);
  }}

  function applyFilters() {{
    const query = search.value.trim().toLowerCase();
    const stage = stageFilter.value;
    visibleLinks = links.filter(link => {{
      const matchesQuery = !query || link.dataset.search.includes(query);
      const stages = (link.dataset.stages || '').split(/\\s+/);
      const matchesStage = !stage || stages.includes(stage);
      const visible = matchesQuery && matchesStage;
      link.hidden = !visible;
      return visible;
    }});
    scoreGroups.forEach(group => {{
      const hasVisiblePhrase = Array.from(group.querySelectorAll('.trajectory-link')).some(link => !link.hidden);
      group.hidden = !hasVisiblePhrase;
      if (query && hasVisiblePhrase) group.open = true;
    }});
    resultCount.textContent = `${{visibleLinks.length}} / ${{links.length}} trajectories`;
    empty.style.display = visibleLinks.length ? 'none' : 'block';
    if (!visibleLinks.length) {{
      panels.forEach(p => p.classList.remove('active'));
      activeId = null;
      return;
    }}
    if (!activeId || !visibleLinks.some(link => link.dataset.target === activeId)) {{
      activate(visibleLinks[0].dataset.target);
    }}
  }}

  function step(delta) {{
    if (!visibleLinks.length) return;
    const currentIndex = Math.max(0, visibleLinks.findIndex(x => x.dataset.target === activeId));
    const nextIndex = Math.min(visibleLinks.length - 1, Math.max(0, currentIndex + delta));
    activate(visibleLinks[nextIndex].dataset.target);
    window.scrollTo({{top: 0, behavior: 'smooth'}});
  }}

  links.forEach(link => link.addEventListener('click', () => activate(link.dataset.target)));
  search.addEventListener('input', applyFilters);
  stageFilter.addEventListener('change', applyFilters);

  panels.forEach(panel => {{
    panel.querySelector('.prev-btn')?.addEventListener('click', () => step(-1));
    panel.querySelector('.next-btn')?.addEventListener('click', () => step(1));
    panel.querySelector('.expand-btn')?.addEventListener('click', () => {{
      panel.querySelectorAll('details.sample, details.block, details.tools').forEach(d => d.open = true);
    }});
    panel.querySelector('.collapse-btn')?.addEventListener('click', () => {{
      panel.querySelectorAll('details.sample, details.block, details.tools').forEach(d => d.open = false);
    }});
  }});

  const hashTarget = location.hash.startsWith('#traj-') ? location.hash.slice(1) : null;
  applyFilters();
  if (hashTarget && links.some(link => link.dataset.target === hashTarget)) activate(hashTarget, false);
}})();
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="one or more messages_train.jsonl files",
    )
    parser.add_argument("-o", "--output", type=Path, help="output HTML path")
    parser.add_argument("--sample-id", action="append", help="render only selected sample IDs")
    parser.add_argument(
        "--companion-input",
        type=Path,
        action="append",
        default=[],
        help="optional prior-stage messages JSONL; kept for compatibility with the old script",
    )
    parser.add_argument(
        "--companion-stage",
        action="append",
        default=[],
        help="only include these agent stages from companion inputs",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="include tool schemas; by default the report shows messages/tool calls only",
    )
    parser.add_argument(
        "--qwen35-template",
        action="store_true",
        help="render the official Qwen3.5-9B serialized training prompt",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        action="append",
        default=[],
        help="private audit JSONL; may be supplied multiple times; defaults to each input directory",
    )
    args = parser.parse_args()

    all_inputs = [*args.inputs, *args.companion_input]
    wanted = set(args.sample_id or [])
    companion_stages = set(args.companion_stage or [])
    samples = load_jsonl(all_inputs, wanted, companion_stages, primary_count=len(args.inputs))
    if not samples:
        raise SystemExit("没有读到符合条件的 sample。")

    if args.audit:
        audit_paths = args.audit
    else:
        audit_paths = [path.with_name("teacher_trajectory_audit.jsonl") for path in all_inputs]
    private_by_id = load_audits(audit_paths)

    output = args.output or args.inputs[0].with_name(args.inputs[0].stem + "_multi.html")
    page = build_page(
        input_paths=all_inputs,
        samples=samples,
        private_by_id=private_by_id,
        show_tools=args.show_tools,
        qwen35_template=args.qwen35_template,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

    groups = group_trajectories(samples)
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in all_inputs],
                "output": str(output),
                "trajectories": len(groups),
                "samples": len(samples),
                "audit_samples": len(private_by_id),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
