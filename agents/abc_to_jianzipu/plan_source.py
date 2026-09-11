from __future__ import annotations

from typing import Any

from .jianzi_renderer import hui_label, render_jianzi_surface


NOTE_NAMES = ("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")
DURATIONS = {60: "六十四分", 120: "三十二分", 240: "十六分", 480: "八分",
             960: "四分", 1920: "二分", 3840: "全音"}


def pitch_label(midi: int | None) -> str:
    if midi is None:
        return "休止"
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _surface_jianzi(action: dict[str, Any]) -> str:
    return render_jianzi_surface(action)


def render_plan_source(score: dict[str, Any], plan: dict[str, Any], *, revision: int,
                       diagnostics: list[dict[str, Any]] | None = None) -> str:
    actions = {source_id: action for action in plan.get("actions", [])
               for source_id in action.get("source_event_ids", [])}
    diagnostics = diagnostics or []
    diagnostic_by_event: dict[str, list[dict[str, Any]]] = {}
    for diagnostic in diagnostics:
        diagnostic_by_event.setdefault(str(diagnostic.get("event_id", "")), []).append(diagnostic)
    lines = [
        f'谱名｜{score.get("title", "未命名")}',
        f'版本｜r{revision:04d}',
        f'调性｜{score.get("key", "")}',
        f'拍号｜{score.get("meter", "")}',
        f'调弦｜{score.get("tuning_name", "")}｜{score.get("open_midi", [])}',
        "",
        "序号@事件ID｜目标音｜时值｜动作｜左手｜右手／续音｜技法｜谱面减字｜诊断",
    ]
    section_start = {section.get("start_event"): section for section in score.get("sections", [])}
    for number, event in enumerate(score.get("events", []), 1):
        section = section_start.get(event["id"])
        if section:
            lines.extend(["", f'〈{section.get("label", "")} {section.get("title", "")}〉'.rstrip()])
        action = actions.get(event["id"])
        if event.get("kind") == "rest":
            action_text, left, right, techniques, full = "休止", "无", "无", "无", "[—]"
        elif action is None:
            action_text, left, right, techniques, full = "未完成", "?", "?", "?", "[?]"
        else:
            mode = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}.get(
                action.get("mode"), str(action.get("mode") or "未知"))
            action_text = mode if action.get("attack") else "续音"
            if action.get("mode") == "stopped":
                finger = action.get("left_finger") or "按指待定"
                left = f'{finger}，按{action.get("string")}弦{hui_label(action.get("hui"))}'
            elif action.get("mode") == "harmonic":
                left = f'{action.get("string")}弦{hui_label(action.get("hui"))}'
            else:
                left = "无"
            right = (f'起音：{action.get("right_finger") or "右手待定"}{action.get("string")}弦'
                     if action.get("attack") else
                     f'续音：承接{action.get("attack_source_id") or "前音"}')
            techniques = "、".join(action.get("techniques") or []) or "无"
            full = _surface_jianzi(action)
        event_diagnostics = diagnostic_by_event.get(event["id"], [])
        diagnostic_text = "；".join(
            f'{item.get("severity", "info")} {item.get("code", "")}' for item in event_diagnostics
        )
        duration = DURATIONS.get(int(event.get("duration_ticks", 0)),
                                 f'{event.get("duration_ticks", 0)} ticks')
        lines.append(
            f'{number:03d}@{event["id"]}｜{pitch_label(max(event.get("pitches_midi") or [None]))}'
            f'｜{duration}｜{action_text}｜{left}｜{right}｜{techniques}｜{full}｜{diagnostic_text}'
        )
    return "\n".join(lines) + "\n"


def render_diagnostics(source_name: str, score: dict[str, Any],
                       diagnostics: list[dict[str, Any]]) -> str:
    line_by_event = {event["id"]: number + 8 for number, event in enumerate(score.get("events", []))}
    lines = []
    for item in diagnostics:
        line = line_by_event.get(item.get("event_id"), 1)
        severity = item.get("severity", "info")
        lines.append(f'{source_name}:{line}:1  {severity} {item.get("code", "UNKNOWN")}')
        details = {key: value for key, value in item.items()
                   if key not in {"severity", "code", "event_id"}}
        if details:
            lines.append(f'  {details}')
    errors = sum(item.get("severity") == "error" for item in diagnostics)
    warnings = sum(item.get("severity") == "warning" for item in diagnostics)
    infos = len(diagnostics) - errors - warnings
    lines.append(f'Build finished: {errors} error(s), {warnings} warning(s), {infos} info(s)')
    return "\n".join(lines) + "\n"
