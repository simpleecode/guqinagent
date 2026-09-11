from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import TaskEnvelope
from .compiler import hui_label
from .pitch_candidates import _pitch
from .plan_editor import apply_plan_edits
from .serialization import score_from_dict
from .skills import SkillRegistry


def _render_plan_edit_preview(actions: list[dict[str, Any]], operations: list[dict[str, Any]],
                              ok: bool, errors: list[dict[str, Any]]) -> str:
    action_ids = [str(operation.get("action_id")) for operation in operations]
    by_id = {str(action.get("action_id")): action for action in actions}
    mode_names = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
    lines = [f'预览｜{"可应用" if ok else "不可应用"}｜变更{len(set(action_ids))}动作',
             "动作ID｜事件｜动作｜左手｜右手／续音｜技法"]
    for action_id in dict.fromkeys(action_ids):
        action = by_id.get(action_id)
        if action is None:
            lines.append(f"{action_id}｜-｜未找到动作｜-｜-｜-")
            continue
        mode = mode_names.get(action.get("mode"), action.get("mode") or "待定")
        position = f'{action.get("string")}弦'
        if action.get("mode") != "open" and action.get("hui") is not None:
            position += hui_label(action.get("hui"))
        events = ",".join(action.get("source_event_ids") or []) or "-"
        lines.append(
            f'{action_id}｜{events}｜{mode}{position}｜{action.get("left_finger") or "无／待定"}｜'
            f'{action.get("right_finger") or ("续音" if not action.get("attack") else "待定")}｜'
            f'{"、".join(action.get("techniques") or []) or "无"}'
        )
    if errors:
        lines.append("错误｜" + str(errors))
    return "\n".join(lines)


TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "list_context": {
        "name": "list_context",
        "description": "列出当前 phrase 之前可按需展开的 phrase 目录；不返回演奏动作。",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "expand_context": {
        "name": "expand_context",
        "description": "展开只读的更早 phrase 序列；不修改当前方案。",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_ref": {"type": "string"},
                "phrase_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "edit_plan": {
        "name": "edit_plan",
        "description": (
            "对当前演奏方案提出带版本前提的局部编辑，并返回确定性验证预览。"
            "工具不直接提交共享状态；Agent 最终输出相同 operations 后由编排器提交。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_revision": {"type": "integer", "minimum": 0},
                "operations": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["base_revision", "operations"],
            "additionalProperties": False,
        },
    },
    "assess_retuning": {
        "name": "assess_retuning",
        "description": "First assess standard tuning resources, then decide whether retuning has enough benefit.",
        "input_schema": {
            "type": "object", "properties": {
                "tonic_midi": {"type": ["number", "null"]},
                "performance_context": {"type": "string", "enum": ["solo", "ensemble"]},
                "open_tone_demand": {"type": "string", "enum": ["auto", "low", "high"]}
            }, "additionalProperties": False,
        },
    },
    "compare_tunings": {
        "name": "compare_tunings",
        "description": "Compare common tunings for this melody using deterministic metrics.",
        "input_schema": {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False,
        },
    },
    "get_pitch_candidates": {
        "name": "get_pitch_candidates",
        "description": (
            "以中文候选表查询某个乐谱事件的可用弦徽位置，作为参考而非白名单。"
            "允许在候选之外提出方案，但仍须计算和审计音高。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "event_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "类型": {"type": "string", "enum": ["泛音", "散音", "按音"]},
                "tolerance_cents": {"type": "number", "minimum": 1, "maximum": 100},
                "modes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["stopped", "open", "harmonic"]},
                },
                "max_candidates": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "anyOf": [{"required": ["event_id"]}, {"required": ["event_ids"]}],
            "additionalProperties": False,
        },
    },
    "calculate_guqin_pitch": {
        "name": "calculate_guqin_pitch",
        "description": (
            "精确计算任意弦、徽位和取音方式的实际 MIDI 音高。"
            "可验证候选集以外的方案。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "string": {"type": "integer", "minimum": 1, "maximum": 7},
                "hui": {"type": ["number", "null"], "minimum": 1, "maximum": 13},
                "mode": {"type": "string", "enum": ["stopped", "open", "harmonic"]},
                "event_id": {"type": "string"},
            },
            "required": ["string", "mode"],
            "additionalProperties": False,
        },
    },
}


ROLE_TOOL_ALLOWLIST = {
    "tuning_planner": {"assess_retuning", "compare_tunings"},
    "phrase_planner": {"expand_context", "get_pitch_candidates", "calculate_guqin_pitch"},
    "guqinization": {"list_context", "expand_context", "get_pitch_candidates", "calculate_guqin_pitch", "edit_plan"},
    "fingering_agent": {"list_context", "expand_context", "get_pitch_candidates", "calculate_guqin_pitch", "edit_plan"},
    "fingering_optimizer": {"list_context", "expand_context", "get_pitch_candidates", "calculate_guqin_pitch", "edit_plan"},
    "style_critic": {"expand_context", "calculate_guqin_pitch"},
}


@dataclass(slots=True)
class ToolBroker:
    """Role-scoped bridge from model tool calls to deterministic skills."""

    registry: SkillRegistry
    traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def definitions_for(self, role: str) -> list[dict[str, Any]]:
        allowed = ROLE_TOOL_ALLOWLIST.get(role, set())
        return [TOOL_DEFINITIONS[name] for name in sorted(allowed)]

    def invoke(self, task: TaskEnvelope, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = ROLE_TOOL_ALLOWLIST.get(task.agent_role, set())
        if name not in allowed:
            result = {"ok": False, "error": "tool_not_allowed_for_role"}
        else:
            try:
                result = {"ok": True, "result": self._invoke_allowed(task, name, arguments)}
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.traces.setdefault(task.task_id, []).append({
            "tool": name,
            "arguments": arguments,
            "result": result,
        })
        return result

    def trace_for(self, task_id: str) -> list[dict[str, Any]]:
        return list(self.traces.pop(task_id, []))

    def _score(self, task: TaskEnvelope):
        raw = task.payload.get("canonical_score")
        if not isinstance(raw, dict):
            raise ValueError("task payload has no canonical_score")
        return score_from_dict(raw)

    def _invoke_allowed(self, task: TaskEnvelope, name: str,
                        arguments: dict[str, Any]) -> dict[str, Any]:
        score = self._score(task)
        if name == "list_context":
            current_phrase_id = str(
                (task.payload.get("phrase_handoff") or {}).get("phrase_id")
                or task.payload.get("phrase_id") or ""
            )
            phrase_ids = [phrase.id for phrase in score.phrases]
            end = phrase_ids.index(current_phrase_id) if current_phrase_id in phrase_ids else len(phrase_ids)
            return {"count": end, "phrases": [
                {"phrase_id": phrase.id, "section_id": phrase.section_id,
                 "start_bar": phrase.start_bar, "end_bar": phrase.end_bar}
                for phrase in score.phrases[:end]
            ], "readonly": True}
        if name == "expand_context":
            phrase_ids = [phrase.id for phrase in score.phrases]
            requested = arguments.get("phrase_id")
            context_ref = arguments.get("context_ref")
            if requested:
                selected_ids = [str(requested)]
            elif context_ref:
                tail = str(context_ref).rstrip("/").split("/")[-1]
                start_id, separator, end_id = tail.partition("-")
                if not separator:
                    selected_ids = [start_id]
                else:
                    start = phrase_ids.index(start_id)
                    end = phrase_ids.index(end_id)
                    selected_ids = phrase_ids[start:end + 1]
            else:
                raise ValueError("context_ref or phrase_id is required")
            unknown = [phrase_id for phrase_id in selected_ids if phrase_id not in phrase_ids]
            if unknown:
                raise ValueError(f"unknown phrase_id: {','.join(unknown)}")
            current_phrase_id = str(
                (task.payload.get("phrase_handoff") or {}).get("phrase_id")
                or task.payload.get("phrase_id") or ""
            )
            if current_phrase_id in phrase_ids:
                current_position = phrase_ids.index(current_phrase_id)
                if any(phrase_ids.index(phrase_id) >= current_position for phrase_id in selected_ids):
                    raise ValueError("expand_context only permits earlier phrases")
            event_ids = {
                event_id
                for phrase in score.phrases if phrase.id in selected_ids
                for event_id in phrase.event_ids
            }
            plan = task.payload.get("performance_plan") or {}
            actions = [
                action for action in plan.get("actions", [])
                if any(event_id in event_ids for event_id in action.get("source_event_ids", []))
            ]
            return {
                "phrase_ids": selected_ids,
                "actions": actions,
                "readonly": True,
            }
        if name == "edit_plan":
            plan = task.payload.get("performance_plan")
            if not isinstance(plan, dict):
                raise ValueError("task payload has no performance_plan")
            revision = int(task.payload.get("plan_revision", 0))
            result = apply_plan_edits(
                plan, current_revision=revision,
                base_revision=int(arguments["base_revision"]),
                operations=list(arguments["operations"]),
            )
            changed_actions = [
                action for action in result["plan"].get("actions", [])
                if any(op.get("action_id") == action.get("action_id")
                       for op in arguments["operations"])
            ]
            return {
                "format": "guqin-edit-preview-1.0",
                "valid": bool(result["ok"]),
                "commit_required": bool(result["ok"]),
                "revision": result.get("revision"),
                "applied_operations": result.get("applied_operations") or [],
                "errors": result.get("errors") or [],
                "text": _render_plan_edit_preview(
                    changed_actions, list(arguments["operations"]),
                    bool(result["ok"]), result.get("errors") or []),
            }
        if name == "assess_retuning":
            return self.registry.call(
                "guqin.decide_tuning", score,
                tonic_midi=arguments.get("tonic_midi"),
                performance_context=arguments.get("performance_context", "solo"),
                open_tone_demand=arguments.get("open_tone_demand", "auto"),
            )
        if name == "compare_tunings":
            names = arguments.get("names")
            return {
                "candidates": self.registry.call("guqin.rank_tunings", score, names=names),
                "advisory_only": True,
            }
        if name == "get_pitch_candidates":
            event_ids = ([str(arguments["event_id"])] if arguments.get("event_id") is not None
                         else [str(value) for value in arguments.get("event_ids") or []])
            if not event_ids:
                raise ValueError("event_id or event_ids is required")
            events = {event.id: event for event in score.events if event.id in set(event_ids)}
            unknown = [event_id for event_id in event_ids if event_id not in events]
            if unknown:
                raise ValueError(f"unknown event_id: {','.join(unknown)}")
            generated = self.registry.call(
                "guqin.pitch_candidates", score,
                tolerance_cents=float(arguments.get("tolerance_cents", 35.0)),
                modes=(({"泛音": "harmonic", "散音": "open", "按音": "stopped"}[
                    arguments["类型"]],) if arguments.get("类型") else
                       tuple(arguments.get("modes", ["stopped", "open", "harmonic"]))),
                max_candidates=int(arguments.get("max_candidates", 18)),
            )
            mode_names = {"open": "散音", "stopped": "按音", "pressed": "按音",
                          "harmonic": "泛音"}
            confidence_names = {"exact": "精确", "approximate": "近似"}
            tables, tops = [], []
            for event_id in event_ids:
                event = events[event_id]
                rows = [item.to_dict() for item in generated.get(event_id, [])]
                target = max(event.pitches_midi or [0])
                lines = [f"目标｜{event_id}｜MIDI {target:g}",
                         "序号｜方式｜弦徽｜实得MIDI｜可信度"]
                for rank, candidate in enumerate(rows, 1):
                    position = f'{candidate.get("string")}弦'
                    if candidate.get("mode") != "open" and candidate.get("hui") is not None:
                        position += hui_label(candidate.get("hui"))
                    confidence = confidence_names.get(
                        candidate.get("confidence"), candidate.get("confidence") or "未知")
                    lines.append(
                        f'{rank}｜{mode_names.get(candidate.get("mode"), candidate.get("mode"))}｜'
                        f'{position}｜{float(candidate.get("sounding_midi")):.1f}｜{confidence}'
                    )
                tables.append("\n".join(lines))
                top = rows[0] if rows else None
                tops.append({"event_id": event_id, "candidate": (
                    {key: top.get(key) for key in
                     ("mode", "string", "hui", "sounding_midi")}
                    if top else None)})
            response = {
                "format": "guqin-candidate-table-1.0",
                "text": "\n\n".join(tables),
                "top_candidates": tops,
                "advisory_only": True,
            }
            if len(tops) == 1:
                response["top_candidate"] = tops[0]["candidate"]
            return response
        if name == "calculate_guqin_pitch":
            string = int(arguments["string"])
            if not 1 <= string <= 7:
                raise ValueError("string must be 1..7")
            mode = str(arguments["mode"])
            hui = arguments.get("hui")
            hui = float(hui) if hui is not None else None
            sounding = _pitch(score.open_midi[string - 1], hui, mode)
            result: dict[str, Any] = {
                "string": string,
                "hui": hui,
                "mode": mode,
                "sounding_midi": round(sounding, 6),
            }
            event_id = arguments.get("event_id")
            if event_id:
                event = next((item for item in score.events if item.id == event_id), None)
                if event is None or event.midi is None:
                    raise ValueError(f"unknown or unpitched event_id: {event_id}")
                result.update({
                    "event_id": event_id,
                    "target_midi": event.midi,
                    "cents_error": round((sounding - event.midi) * 100, 3),
                })
            return result
        raise KeyError(name)
