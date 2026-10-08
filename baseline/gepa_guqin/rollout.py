"""Single-stage public agent rollout on the deterministic tool runtime.

Mirrors ``train/scripts/eval_two_stage_score.py::run_stage_steps`` for the
``single_stage`` branch (blank plan, warning injection, pitch-warning
follow-up gate, repeated-submission acceptance, per-attempt retry), but the
model turn goes to an Anthropic-compatible GLM backend with native
``tool_use`` blocks instead of a local Qwen/vLLM generation.

The system prompt is a *parameter*: GEPA candidates replace the production
``public_system_for("single_stage")`` text, everything else (tool schemas,
user prompt, gates) stays fixed.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    blank_plan_from_item, public_tools_for,
)
from agents.abc_to_jianzipu.teacher_trajectory import render_public_prompt  # noqa: E402
from agents.ToolRuntime import (  # noqa: E402
    RealToolRuntime, harmonic_region_at_phrase_start,
    public_pitch_warning_source_indices, public_plan_warning_messages,
    validate_jianzi_only,
)
from agents.abc_to_jianzipu.trajectory_replay import replay_patches  # noqa: E402
from ABC_J.scripts.generate_teacher_tool_trajectories import canonical_jianzi_text  # noqa: E402

from .glm_client import (  # noqa: E402
    GlmBackend, history_blocks, tool_calls, truncated, visible_text,
)
from .compat import action_text, materialize_actions  # noqa: E402

STAGE = "single_stage"


class ScoreRunState:
    """Sequential per-score driver state: candidate-self handoff only.

    Mirrors ``eval_two_stage_score.py``'s serial loop exactly: the read-only
    previous phrase injected into the next user prompt, the phrase history
    exposed to context tools, and the harmonic-region flag all come from the
    *model's own* finalized plans for this run — never from the reference.
    A failed phrase poisons the rest of its score (later phrases must not see
    an incomplete or gold predecessor).
    """

    def __init__(self):
        self.previous: dict | None = None
        self.historical: dict[tuple[str, str], dict] = {}

    def prepare(self, working_item: dict) -> dict | None:
        """Inject the model-own handoff + harmonic flag; returns the handoff."""
        handoff = working_item["input"].setdefault("phrase_handoff", {})
        handoff.pop("previous_phrase", None)
        injected = None
        if self.previous is not None:
            injected = {
                "phrase_id": self.previous["phrase_id"],
                "status": "confirmed_readonly",
                "notes": self.previous["input"]["notes_without_jianzi"],
                # The model's own preceding output, never the sealed
                # reference annotation.
                "actions": materialize_actions(
                    self.previous["reference_plan"]["actions"]),
            }
            handoff["previous_phrase"] = injected
        working_item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(
            working_item, self.historical
        )
        return injected

    def commit(self, working_item: dict, session: dict) -> bool:
        """Record the finalized model plan; False poisons the score tail."""
        if not session.get("ok"):
            return False
        working_item["reference_plan"] = session["plan"]
        self.previous = working_item
        self.historical[(str(working_item["score_key"]),
                         str(working_item["phrase_id"]))] = working_item
        return True


def _error_followup(warning_followup_required: bool) -> dict[str, str]:
    if warning_followup_required:
        error = ("上一轮出现音高警告；请至少提交一次 edit_plan 尝试后再决定"
                 "是否结束。请依据上一轮工具返回核对相关音序。")
    else:
        error = "当前段仍有待填写的演奏音，不能直接结束；请提交需要修改的 jianzi_rows。"
    return {
        "role": "user",
        "content": json.dumps({
            "tool_results": [{"ok": False, "error": error}],
            "instruction": "根据真实反馈继续；需要工具时继续调用工具。",
        }, ensure_ascii=False),
    }


def run_single_stage_phrase(example: dict[str, Any], historical: dict[tuple[str, str], Any],
                            system_text: str, backend: GlmBackend, *,
                            max_rounds: int = 8, attempts: int = 2,
                            sample_temperature: float = 0.25,
                            log: Any = None) -> dict[str, Any]:
    """Run one phrase through the public single-stage loop.

    ``example`` is a prepared data-split row (``runtime_item`` already carries
    the reference handoff and harmonic flag) or an equivalent final-eval row
    (handoff built from the model's own previous phrase).  ``historical`` maps
    ``(score_key, phrase_id)`` to finalized rows for context tools and
    phrase-spanning validation state.
    """
    item = deepcopy(example["runtime_item"])
    item["public_pitch_warning_source_indices"] = sorted(
        public_pitch_warning_source_indices(item, historical=historical)
    )
    item["public_plan_warning_messages_by_source"] = public_plan_warning_messages(
        item, historical=historical
    )
    item["baseline_plan"] = blank_plan_from_item(item)
    user_prompt = render_public_prompt(item, STAGE)
    tools = public_tools_for(STAGE, basic=False)
    source_to_event = {
        int(note["index"]): int(note["event_index"])
        for note in item["input"].get("notes_without_jianzi") or []
        if note.get("index") is not None and note.get("event_index") is not None
    }

    last_trace: list[dict[str, Any]] = []
    for attempt in range(max(1, attempts)):
        runtime = RealToolRuntime(item, historical, basic_fingering=False)
        api_messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        trace: list[dict[str, Any]] = []
        warning_followup_required = False
        saw_pitch_warning = False
        last_valid: list[dict[str, Any]] | None = None
        for round_number in range(1, max_rounds + 1):
            round_entry: dict[str, Any] = {
                "round": round_number, "raw_output": "", "truncated": False,
                "tool_calls": [], "tool_results": [],
            }
            try:
                response = backend.generate(
                    system=system_text, messages=api_messages, tools=tools,
                    temperature=(sample_temperature if attempt else 0.0),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                round_entry["generation_error"] = error
                trace.append(round_entry)
                if log:
                    log({"event": "rollout_generation_error",
                         "sample_id": example.get("sample_id"), "round": round_number,
                         "error": error})
                api_messages.append({
                    "role": "user",
                    "content": (
                        "上一轮工具调用未能执行：模型输出的工具参数不是完整合法的 JSON，"
                        f"错误为 {error}。未执行任何工具，也未修改曲谱。"
                        "请重新调用一个工具；参数必须是完整 JSON，尤其确认数组、花括号和引号均已闭合。"
                    ),
                })
                continue
            raw = visible_text(response)
            calls = tool_calls(response)
            round_entry["raw_output"] = raw
            round_entry["truncated"] = truncated(response)
            round_entry["tool_calls"] = [
                {"name": call["name"], "arguments": call["arguments"]} for call in calls
            ]
            trace.append(round_entry)
            if log:
                log({"event": "rollout_round", "sample_id": example.get("sample_id"),
                     "attempt": attempt, "round": round_number,
                     "tool_calls": [call["name"] for call in calls],
                     "truncated": round_entry["truncated"]})
            if not calls:
                api_messages.append({"role": "assistant",
                                     "content": raw or "(无文字输出)"})
                if last_valid is not None and not warning_followup_required:
                    return {"ok": True, "no_op": False,
                            "plan": {"actions": last_valid}, "trace": trace,
                            "final_reply": raw or "工具预览已通过，当前段减字填写完成。",
                            "user_prompt": user_prompt}
                api_messages.append(_error_followup(warning_followup_required))
                continue

            api_messages.append({"role": "assistant", "content": history_blocks(response)})
            tool_result_blocks: list[dict[str, Any]] = []
            accepted: list[dict[str, Any]] | None = None
            warned_now: set[int] = set()
            repeated_submission: list[int] | None = None
            for call in calls:
                name = call["name"]
                arguments = call.get("arguments") or {}
                if name == "edit_plan":
                    submitted = [
                        row for row in arguments.get("jianzi_rows") or []
                        if isinstance(row, list) and len(row) == 2
                        and isinstance(row[0], int) and not isinstance(row[0], bool)
                    ]
                    if submitted:
                        current = {
                            int(action["source_index"]): action_text(action)
                            for action in replay_patches(
                                item["baseline_plan"], runtime.accumulated_patches,
                                strict_before=False,
                            ).actions
                        }
                        unchanged_events: list[int] = []
                        for event_index, text in submitted:
                            try:
                                source_index = runtime._event_to_source(event_index)
                            except ValueError:
                                break
                            if canonical_jianzi_text(text) == canonical_jianzi_text(
                                current.get(source_index)
                            ):
                                unchanged_events.append(event_index)
                        if len(unchanged_events) == len(submitted):
                            repeated_submission = unchanged_events
                result = runtime.invoke(name, arguments)
                round_entry["tool_results"].append({
                    "name": name, "arguments": arguments, "result": result,
                })
                tool_result_blocks.append({
                    "type": "tool_result", "tool_use_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
                if name != "edit_plan":
                    continue
                if result.get("ok") and result.get("result", {}).get("valid"):
                    complete, report = validate_jianzi_only(
                        item, runtime.accumulated_patches,
                        toward_reference=False, require_complete=True,
                        historical=historical,
                    )
                    if complete:
                        accepted = materialize_actions(
                            runtime.calls[-1]["result"]["result"]["preview_actions"]
                        )
                        warned_now = {
                            source_to_event.get(int(warning["source_index"]),
                                                int(warning["source_index"]))
                            for warning in report.get("warnings") or []
                            if warning.get("code") == "jianzi_pitch_mismatch"
                            and warning.get("source_index") is not None
                        }
            api_messages.append({"role": "user", "content": tool_result_blocks})
            if accepted is not None:
                last_valid = accepted
                if repeated_submission:
                    round_entry["repeated_submission"] = sorted(repeated_submission)
                    return {
                        "ok": True, "no_op": False,
                        "plan": {"actions": accepted}, "trace": trace,
                        "final_reply": ("检测到重复提交未改变的音序（"
                                        + "、".join(str(index) for index in sorted(repeated_submission))
                                        + "），采用当前有效预览并停止交互。"),
                        "user_prompt": user_prompt,
                    }
                if warning_followup_required:
                    warning_followup_required = False
                if warned_now and not saw_pitch_warning:
                    saw_pitch_warning = True
                    warning_followup_required = True
                round_entry["pitch_gate"] = {
                    "warned_now": sorted(warned_now),
                    "followup_edit_required": warning_followup_required,
                }
                continue
        # Rounds exhausted: keep the last valid preview rather than failing the
        # phrase outright (a broken stage would poison the cascade).
        if last_valid is not None:
            return {"ok": True, "no_op": False,
                    "plan": {"actions": last_valid}, "trace": trace,
                    "final_reply": "轮次用尽，采用最后一份有效预览。",
                    "pitch_followup_edit_required_at_stop": warning_followup_required,
                    "user_prompt": user_prompt}
        last_trace = trace
    return {"ok": False, "trace": last_trace, "user_prompt": user_prompt}
