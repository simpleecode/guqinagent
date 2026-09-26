#!/usr/bin/env python3
"""Generate answer-guided trajectories whose observable tools are really executed."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # Keep the generator usable in minimal inference envs.
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    from dotenv import load_dotenv
except ImportError:  # Evaluation/runtime-only environments do not need .env loading.
    def load_dotenv(*_args, **_kwargs):
        return False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.inverse_baseline import AUDIT  # noqa: E402
from agents.abc_to_jianzipu.jianzi_renderer import (  # noqa: E402
    hui_label, render_jianzi_surface,
)
from agents.abc_to_jianzipu.inverse_models import ReferenceAction  # noqa: E402
from agents.abc_to_jianzipu.models import ScoreEvent  # noqa: E402
from agents.abc_to_jianzipu.pitch_candidates import (  # noqa: E402
    abc_chord_midis, abc_chord_tones, candidates_for_event,
)
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER  # noqa: E402
from agents.abc_to_jianzipu.teacher_trajectory import (  # noqa: E402
    _public_input, _render_phrase_lines, chord_jianpu_label, render_public_prompt,
)
from agents.abc_to_jianzipu.trajectory_replay import (  # noqa: E402
    compare_replay_to_patch_targets, replay_patches,
)
from agents.ToolRuntime import (  # noqa: E402
    RealToolRuntime, advance_pitch_warning_state, can_accept_empty_tool_turn,
    harmonic_region_at_phrase_start, infer_jianzi_text_patches,
    public_pitch_warning_source_indices, validate_jianzi_only,
)
from agents.ToolRuntime.runtime import (  # noqa: E402
    canonical_jianzi_text, compact_context_catalog, event_index_for_source,
    expand_jianzi_rows, infer_jianzi_text_patches,
    is_all_empty_reference_phrase, normalize_patches, pitch_audit_notes,
    pitch_label, render_edit_preview, render_grouped_candidate_table,
    source_index_for_event, target_midis, validate_patch_vocabulary,
)

BANNED_TEXT = ("teacher_private", "reference_plan", "reference_actions", "target_patches",
               "标准答案", "参考答案", "私有答案")


BANNED_TEXT += ("reference action", "reference plan", "teacher-only", "private answer",
                "参考答案", "标注答案", "私有答案", "教师提示", "系统提示", "提示词",
                "教师规则", "私有规则", "与参考一致", "参考使用", "参考中", "按提示")
BANNED_TEXT += ("GQS", "教师私有", "最终标注")

NOOP_CONCLUSION = "综上所述，无需修改。"
PURE_CUO_RULE = (
    "普通撮本身就是右手指法：括号内只写参与的两弦及各弦必要的散音、按音、左手按指和徽位信息，不再写“勾、挑、托”等右手指法；也不得把“撮”与二至三个中文数字直接拼成缩略写法。"
)


def validate_normalized_tuning(item: dict) -> None:
    """Reject stale inferred seeds before any teacher API call.

    The frozen ``normalized_tuning`` field is consumed by candidate lookup and
    edit-plan auditing.  It must exactly reproduce the sounding open strings
    in source metadata; otherwise a historic double-application of tuning
    offsets can silently make every candidate table wrong.
    """
    input_data = item.get("input") or {}
    actual = (input_data.get("normalized_tuning") or {}).get("open_midi")
    expected = AUDIT.parse_open_midi(input_data.get("metadata") or {})
    if (not isinstance(actual, list) or len(actual) != 7
            or [float(value) for value in actual] != [float(value) for value in expected]):
        raise ValueError(
            "normalized_tuning mismatch for "
            f"{item.get('trajectory_id')}: actual={actual!r}, expected={expected!r}"
        )


def finalize_noop_review(text: str) -> str:
    """Give every accepted no-op review one unambiguous stopping sentence."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    # Avoid duplicating a semantically identical conclusion emitted by the
    # teacher while keeping its preceding musical analysis unchanged.
    value = re.sub(
        r"(?:综上所述[，,]?)?(?:当前稿|当前谱面|本段)?(?:无需|不需)修改[。.!！]*$",
        "",
        value,
    ).rstrip(" ；;。.!！")
    return f"{value}。{NOOP_CONCLUSION}" if value else NOOP_CONCLUSION

def fingering_complete(plan: dict) -> bool:
    return bool(plan.get("actions")) and all(
        action.get("jianzi_text") is not None
        for action in plan.get("actions", [])
    )


def blank_fingering_plan(plan: dict) -> dict:
    """Keep deterministic event coverage while removing Route Search choices."""
    actions = []
    for source in plan.get("actions", []):
        action = deepcopy(source)
        for field in ("mode", "string", "hui", "left_finger", "right_finger"):
            action[field] = None
        action["compound_gesture"] = None
        action["techniques"] = []
        action["pre_attack_techniques"] = []
        action["jianzi_text"] = None
        action["evidence"] = ["FINGERING_FROM_ZERO"]
        actions.append(action)
    return {"actions": actions}


def blank_plan_from_item(item: dict) -> dict:
    """Blank scaffold from Route Search coverage when present, otherwise one
    blank action per note row. Bars are structural; ordinary notes, rests and
    sustains remain text-editable because a rest can carry continuation
    notation such as 走猱. Phrases whose melody needs 散／泛 have no
    stopped-only baseline route by construction and must not be dropped."""
    base = item.get("baseline_plan") or {}
    if base.get("actions"):
        plan = blank_fingering_plan(base)
        actions = plan["actions"]
        covered = {int(action["source_index"]) for action in actions}
        for note in item["input"].get("notes_without_jianzi", []):
            index = int(note["index"])
            abc = str(note.get("abc") or "")
            jianpu = str(note.get("jianpu") or "")
            # The route-search baseline can legitimately omit a normal note
            # (for example when its pitch is ambiguous).  It is still a
            # text-editable score row: the teacher may know a valid jianzi
            # even when the structural fields cannot be inferred.
            # A bar is structural and never carries a jianzi.  A rest or
            # sustain, however, can carry continuation ornaments (e.g.
            # ``0（休止）｜[走猱]``), so it remains text-editable even without
            # a pitch target.
            if index in covered or jianpu.strip() == "|":
                continue
            is_sustain = (abc.startswith("-") or "延音" in jianpu
                          or "休止" in jianpu)
            actions.append({
                "source_index": index, "mode": None, "string": None,
                "hui": None, "left_finger": None, "right_finger": None,
                "techniques": [], "pre_attack_techniques": [],
                "attack": not is_sustain,
                "evidence": (["DISPLAY_ONLY_NONATTACK"] if is_sustain
                             else ["TEXT_ONLY_EDITABLE"]),
                "notation_omitted": False,
                "jianzi_text": "" if is_sustain else None,
            })
        actions.sort(key=lambda action: int(action["source_index"]))
        return plan
    actions = []
    for note in item["input"].get("notes_without_jianzi", []):
        if str(note.get("jianpu") or "").strip() == "|":
            continue
        jianpu = str(note.get("jianpu") or "")
        abc = str(note.get("abc") or "")
        is_sustain = abc.startswith("-") or "延音" in jianpu or "休止" in jianpu
        actions.append({
            "source_index": int(note["index"]), "mode": None, "string": None,
            "hui": None, "left_finger": None, "right_finger": None,
            "techniques": [], "pre_attack_techniques": [], "attack": not is_sustain,
            "evidence": ["FINGERING_FROM_ZERO"],
            "notation_omitted": bool(note.get("notation_omitted")),
            "jianzi_text": "" if is_sustain else None,
        })
    return {"actions": actions}

TOOLS = [
    {"name": "list_context", "description": "按曲谱章节压缩列出当前 phrase 之前可展开的只读上下文 phrase ID；不返回谱面动作或音序范围。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "expand_context", "description": "展开一个未在 user 中提供的更早 phrase 的只读已确认谱面。已直接给出的“只读前一段”不得重复调用。返回与当前谱面相同的简表。",
     "input_schema": {"type": "object", "properties": {
         "phrase_id": {"type": "string"}}, "required": ["phrase_id"]}},
    {"name": "get_pitch_candidates", "description": "查询取音位置候选。参数使用当前段连续的音序；首次查询时，若当前段有4个或以上可解析的发音事件，尽量把至少4个（最好全部）音序一次放入 event_indices；后续只有确需聚焦核查某个音时才单独查询。类型可传一个字符串或字符串数组。工具会解析调号、八度与和弦，并按实际目标 MIDI 自动去重，每个音高只返回一份候选及其全部来源音序。target_midi 仅用于脱离谱面事件的特殊查询。",
     "input_schema": {"type": "object", "properties": {
         "event_index": {"type": "integer"},
         "event_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
         "target_midi": {"type": "number"},
         "类型": {"oneOf": [{"type": "string", "enum": ["泛音", "散音", "按音"]}, {"type": "array", "items": {"type": "string", "enum": ["泛音", "散音", "按音"]}, "minItems": 1}]},
         "max_candidates": {"type": "integer"}},
      "anyOf": [{"required": ["event_index"]}, {"required": ["event_indices"]},
                {"required": ["target_midi"]}]}},
    {"name": "edit_plan", "description": "批量预览并设置减字文本。唯一输入为 jianzi_rows=[[音序,减字文字],...]；音序应为 JSON 整数（例如 [12,\"吟\"]，不要写成 [\"12\",\"吟\"]）。一次调用可以只提交已决定修改的部分行，未提交的行不会使本次预览失败；Fingering 阶段应继续分批填写，直到结束前所有发音事件都有字符串（必要时为空字符串）。空字符串的直接语义是将该行 jianzi_text 置空；不会删除声音或其他演奏状态。若前一减字已包含多个动作，后续被覆盖的声音可用空字符串表示不重复显示。工具只在减字和简谱都可高置信解析且音高明显不符时给出警告、应该尽量修复",
     "input_schema": {"type": "object", "properties": {
             "jianzi_rows": {"type": "array", "description": "本次决定编辑的行 [音序,jianzi_text]；可以只提交部分行，未提交的行保留现状，不会因尚未决定而失败。音序为 JSON 整数，小节线没有音序。空字符串表示将该行 jianzi_text 置空；不会删除声音或演奏状态，常用于前一复合减字已覆盖后续声音、无需重复显示的情况。", "items": {
             "type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}],
             "minItems": 2, "maxItems": 2}},
     }, "required": ["jianzi_rows"],
        "additionalProperties": False}},
]

# New trajectories use one source of truth: the reduced-notation text.  The
# older strings above remain only to keep historical source diffs readable;
# this assignment is the complete public contract used at runtime.
PUBLIC_SYSTEM = {
    "fingering_agent": (
        "你是减字谱初稿 Agent。阅读简谱、ABC、时值和前一段减字，"
        "为当前段每个演奏事件填写基础减字；也可以填写\"再作标记\"/\"再作\"/\"从ㄱ再作\"等表示省略的减字。通过 edit_plan.jianzi_rows 进行编辑；"
        "基础初稿只使用泛音、按音、散音三种取音方式，右手只使用抹、挑、勾、剔、擘、托、打、摘、撮；"
        "填写或编辑曲谱只能使用 edit_plan.jianzi_rows；尽量为当前段每个发音事件填写减字，"
        "暂不主动加入绰、注、吟、猱、走手或其他复杂技法。"
        "若工具在已填写的减字后标注\":warning:音高不匹配\"，应尽量核对并修正该行的取音与减字；若经判断无需改动则保留，并继续完成其他合法编辑。"
    ),
    "guqinization": (
        "你是减字谱润色 Agent。基于初稿、上下文与专业判断，修订走手、复合技法和显示范围。"
        "通过 edit_plan.jianzi_rows 进行编辑；其中只列出相对当前初稿确实需要改写的音，不用重发无需改变的行；"
        "空字符串表示将该行 jianzi_text 置空；不会删除声音或演奏状态。"
        f'"{OMITTED_PLACEHOLDER}"表示该音采用再作动作继承的减字（前一段或再作标记后）、因此谱面省略其减字。若前一减字已覆盖多个动作，后续行可用空字符串避免重复显示。'
        "工具的音高结果是高置信情况下的辅助警告，如果有音高警告，需要尽量修正。"
        "当前段减字由只会基础指法的 Agent 初步填写；请在不破坏音高和可演奏性的前提下适当加入高级指法，使曲子更丰富、更有韵味。"
    ),
    "single_stage": (
        "你是减字谱 Agent。直接从简谱、ABC、时值与上下文，为当前段一次完成可演奏的最终减字谱；"
        "同时判断取音方式、左手按指与徽位、右手指法、走手/复合技法以及必要的空显示范围。"
        "通过 edit_plan.jianzi_rows 进行编辑；可分批提交已决定的行，未提交的行保留现状。"
        "空字符串表示该行不重复显示减字，不删除声音或演奏状态；若前一复合减字已覆盖后续声音可使用。"
        "优先保证音高、左右手可演奏性与前后衔接，再按需要使用绰、注、吟、猱、历、撮等复杂技法，"
        "不要为了堆砌技法而改坏已经合理的取音。"
        "若工具在已填写的减字后标注\":warning:音高不匹配\"，应尽量核对并修正该行；"
        "若结合候选与音乐判断决定保留，也可说明理由后结束。"
    ),
}

INHERITANCE_RULES_PROMPT = (
    "减字谱采用按字段、按动作类型的局部继承：省略左手按指或徽位通常沿用当前左手状态，仅写弦序时右手指法可以承前；“就”沿用走手后到达的当前位置。"
    "吟、猱、撞、绰、注等左手续作通常承接前一按音的弦、按指与当前音位，不表示新的右手起音；泛起至泛止之间继承泛音状态。"
    "明确写出的新按指、徽位、音色或移位动作会更新相应状态，不能因某字段省略就把上一音的全部状态机械照搬。"
)


def public_tools_for(stage: str, *, basic: bool = False) -> list[dict]:
    allowed = {"list_context", "expand_context", "edit_plan"}
    # Guqinizer can also receive a pitch warning after changing a concrete
    # fingering.  It needs the same read-only lookup tool to repair that edit.
    if basic or stage in {"guqinization", "single_stage"}:
        allowed.add("get_pitch_candidates")
    return [deepcopy(tool) for tool in TOOLS if tool["name"] in allowed]


def public_system_for(stage: str, *, basic: bool = False) -> str:
    prompt = PUBLIC_SYSTEM[stage] + (PURE_CUO_RULE if stage == "fingering_agent" and basic else "") + INHERITANCE_RULES_PROMPT + (
        "每次工具调用前说明理由。get_pitch_candidates 与 edit_plan 不得在同一 assistant 回合同时调用：先查询并读取候选工具返回，再在后续回合提交编辑。"
        "edit_plan 一次可提交多行，也可以只提交当前已经确定的部分行；未提交的行保留现状，不会因尚未决定而使本次预览失败。收到反馈后只修正需要变化的行，不要重复未变化内容。"
    )
    if stage in {"guqinization", "single_stage"}:
        prompt += (
            "先逐一分析每个音应该使用什么指法/减字，再决定是否编辑；"
            "只把决定改写的音提交给 edit_plan。"
        )
    if basic:
        completion_scope = "最终减字" if stage == "single_stage" else "基础减字"
        prompt += (
            " 首次查询音高时，若当前段有4个或以上可解析的发音事件，尽量把至少4个（最好全部）音序放在同一次 get_pitch_candidates 的 event_indices 中；工具会按实际音高自动去重。只有后续确需核查某个单音时才逐音查询，不要把首次查询拆成逐音调用。"
            f"依据候选位置、上下文和专业判断写出{completion_scope}。尽量把每个演奏事件的减字填写完整，除非谱面关系很明显需要留空。"
            "可以分批提交已经确定的音，继续编辑直到结束前所有发音事件都有减字或明确空字符串。"
            "休止和延音没有新音高，但若要表达走猱、猱、吟、泛止等延续动作，可以提交对应减字。"
        )
    return prompt






def render_teacher_reference_gqs(item: dict) -> str:
    """Private musical goal in the same table grammar as the editable phrase.

    The complete reference objects remain in the private audit artifact and are
    still used by deterministic validators.  Giving the teacher JSON-like GQS
    rows next to a pipe-table editable phrase makes harmless notation variants
    needlessly hard to compare, so the readable view deliberately mirrors the
    student-visible columns.  Evidence, diagnostics, field masks, null fields,
    and other machine-only bookkeeping stay out of the model context.
    """
    lines = [
        "当前段参考｜只读｜仅作改进方向",
        "序号｜简谱｜ABC｜时值｜谱面减字",
    ]
    references = {
        int(action["source_index"]): action
        for action in item.get("reference_plan", {}).get("actions", [])
    }
    last_marker = None
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        index = int(note["index"])
        section = note.get("section") or {}
        marker = str(section.get("marker") or "")
        # A section marker is emitted only at the actual section boundary.
        # Every note carries the owning section metadata, so checking merely
        # ``marker != last_marker`` would repeat <一>/<二>/... at the start of
        # every phrase that continues an existing section.
        if marker and section.get("start") and marker != last_marker:
            lines.append(marker)
        if marker:
            last_marker = marker
        reference = references.get(index) or {}
        jianzi = str(reference.get("jianzi_text") or reference.get("text") or "")
        # The editable phrase renderer conventionally writes string numbers
        # as Chinese numerals. Keep that harmless surface convention aligned
        # while leaving substantive reference notation untouched.
        jianzi = re.sub(
            r"([1-7])弦",
            lambda match: "一二三四五六七"[int(match.group(1)) - 1] + "弦",
            jianzi,
        )
        # A materialized repeat is not an empty annotation: its glyph is
        # deliberately omitted because the already-written passage is played
        # again.  Keep that distinction visible to the private teacher while
        # the public/current editable phrase still starts from 待填写.
        is_bar = (str(note.get("jianpu") or "").strip() == "|"
                  or str(note.get("abc") or "").strip() == "|")
        if is_bar:
            lines.append("小节线")
            continue
        elif note.get("notation_omitted") and not jianzi:
            surface = f"[{OMITTED_PLACEHOLDER}]"
        elif reference:
            # An ordinary blank annotation is an explicit display target.  It
            # may ask Guqinizer to clear a redundant glyph after 吟/猱 or a
            # multi-sound compound gesture, so never render it as "unknown".
            surface = f"[{jianzi}]" if jianzi else "[空]"
        else:
            # Structural rows such as barlines have no reference action.
            surface = ""
        visible_index = event_index_for_source(item, index)
        visible_index = index if visible_index is None else visible_index
        lines.append("｜".join([
            str(visible_index), pitch_label(item, index), str(note.get("abc") or ""),
            str(note.get("duration") or "-"), surface,
        ]))
    return "\n".join(lines)


# Full source corpus is kept in ``knowledge/complex_fingering_explanations``.
# The loader splits slash-separated aliases and removes duplicate lookup keys.
# The mapping file is curated for injection: every indexed name is eligible,
# including single-character complex gestures such as “逗”.
COMPLEX_FINGERING_KNOWLEDGE_PATH = (
    ROOT / "agents/abc_to_jianzipu/knowledge/complex_fingering_explanations_v3_with_effects.jsonl"
)
def load_complex_gesture_knowledge() -> tuple[
    dict[str, dict[str, str]], dict[str, list[dict[str, str]]]
]:
    """Load every slash alias while retaining all entries sharing an alias."""
    knowledge: dict[str, dict[str, str]] = {}
    matches: dict[str, list[dict[str, str]]] = {}
    for line in COMPLEX_FINGERING_KNOWLEDGE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        explanation = str(row["explanation"]).strip()
        category = str(row["category"]).strip()
        # Entries such as “叠蠲 / 叠涓古写” share one explanation.
        indexed_name = str(row["name"]).strip()
        for name in (part.strip() for part in indexed_name.split("/")):
            if not name:
                continue
            candidate = {
                "indexed_name": indexed_name,
                "category": category,
                "explanation": explanation,
            }
            bucket = matches.setdefault(name, [])
            if candidate not in bucket:
                bucket.append(candidate)
            # Preserve the historical single-entry lookup API for callers that
            # inspect aliases directly; injection uses ``matches`` below and
            # therefore retains every entry (for example 撮 and 大撮/撮).
            knowledge.setdefault(name, {
                "category": category, "explanation": explanation,
            })
    return knowledge, matches


COMPLEX_GESTURE_KNOWLEDGE, COMPLEX_GESTURE_KNOWLEDGE_MATCHES = (
    load_complex_gesture_knowledge()
)

def matched_compound_gesture_knowledge(item: dict) -> list[dict[str, str]]:
    """Return the exact private knowledge entries selected for one phrase."""
    matched: list[str] = []
    reference_text = "\n".join(
        str(action.get("jianzi_text") or action.get("text") or "")
        for action in item.get("reference_plan", {}).get("actions", [])
    )
    eligible = sorted(COMPLEX_GESTURE_KNOWLEDGE, key=len, reverse=True)
    for gesture in eligible:
        if gesture not in reference_text:
            continue
        # “掐撮三声” subsumes “掐撮”; inject only the most specific note.
        if any(gesture in prior for prior in matched):
            continue
        matched.append(gesture)
    details: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for gesture in matched:
        entries = COMPLEX_GESTURE_KNOWLEDGE_MATCHES[gesture]
        for entry in entries:
            identity = (entry["indexed_name"], entry["explanation"])
            if identity in seen:
                continue
            seen.add(identity)
            details.append({
                # A lone alias is labelled with the spelling actually found in
                # the phrase.  If one spelling selects multiple entries, show
                # their full index names so reviewers can distinguish them.
                "name": (gesture if len(entries) == 1 else entry["indexed_name"]),
                "category": entry["category"],
                "explanation": entry["explanation"],
            })
    return details


def render_compound_gesture_knowledge(item: dict) -> list[str]:
    """Private, longest-match knowledge notes for complex reference gestures."""
    return [
        f"复合技法知识｜{detail['name']}：{detail['explanation']}"
        for detail in matched_compound_gesture_knowledge(item)
    ]




def render_annotation_gqs(item: dict) -> str:
    """Lossless phrase-sized annotation view for private audit/visualization."""
    references = {
        int(action["source_index"]): action
        for action in item.get("reference_plan", {}).get("actions", [])
    }
    lines = ["GQS｜annotation-phrase-1.2", "列｜音序｜简谱｜减字"]
    last_marker = None
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        index = int(note["index"])
        section = note.get("section") or {}
        marker = str(section.get("marker") or "")
        # Do not turn the section label attached to ordinary notes into a
        # paragraph marker for every phrase; emit it only at its true start.
        if marker and section.get("start") and marker != last_marker:
            lines.append(marker)
        if marker:
            last_marker = marker
        reference = references.get(index) or {}
        text = str(reference.get("text") or reference.get("jianzi_text") or "")
        is_bar = (str(note.get("jianpu") or "").strip() == "|"
                  or str(note.get("abc") or "").strip() == "|")
        if note.get("notation_omitted") and not text and not is_bar:
            text = OMITTED_PLACEHOLDER
        if is_bar:
            lines.append("小节线")
            continue
        row = [event_index_for_source(item, index) if event_index_for_source(item, index) is not None else index,
               pitch_label(item, index), text]
        lines.append("音｜" + json.dumps(
            row, ensure_ascii=False, separators=(",", ":")
        ))
    return "\n".join(lines)


def render_teacher_system(instruction: dict, reference_gqs: str | None = None) -> str:
    """Keep readable score lines out of nested JSON string escaping."""
    system = json.dumps(instruction, ensure_ascii=False)
    if reference_gqs:
        system += "\n\n【教师私有参考谱】\n" + reference_gqs
    return system


def private_reference_semantics_rules(stage: str) -> list[str]:
    """Explain private reference display markers without leaking them publicly."""
    harmonic_scope_rule = (
        "泛起至泛止构成泛音区间：区间内未显式写“散”或“按音”的起音默认按泛音解释，"
        "泛音区间内不必每音重复写“泛音”。例如“大指七徽挑七弦”等同于“泛音大指七徽挑七弦”。"
        "显式“散”只覆盖当前音，不结束后续泛音区间；只有“泛止”才结束该区间。"
        "决定要进入泛音区间的话，请给出理由"
    )
    warning_rule = (
        '工具在已填写的减字后标注":warning:音高不匹配"时，应尽量核对该行的取音与减字并修正；'
        '这是非阻塞警告，处理该行后继续完成其他合法编辑。'
    )
    if stage == "fingering_agent":
        return [
            harmonic_scope_rule,
            f'私有参考表中的"{OMITTED_PLACEHOLDER}"表示再作谱面省略，不是空标注，也不是未知值；该音仍有演奏语义，不得仅因该标记删除声音；该标记本身也是允许提交的减字文字，可通过 edit_plan.jianzi_rows 直接填写"{OMITTED_PLACEHOLDER}"；这表示沿用再作动作继承的减字（前一段或标记处），不等于把该音置为空字符串。',
            warning_rule,
        ]
    return [
        harmonic_scope_rule,
        "私有参考表中的[空]是确定的谱面空显示目标，不表示缺少资料。"
        "若当前初稿在该行有减字，而前一吟、猱、走手或复合多声减字已覆盖该动作，"
        "应在 edit_plan.jianzi_rows 中把该行减字设为空字符串；置空只隐藏本行减字，不删除声音或演奏状态。",
        f'若当前初稿已有"{OMITTED_PLACEHOLDER}"且与私有参考一致，必须保留该文字，不得改为空字符串；若私有参考要求其他文字，再按逐音分析决定。',
        warning_rule,
    ]


def private_only_reference_phrases(item: dict) -> tuple[str, ...]:
    """Return nonempty reference texts not already visible in the current draft."""
    baseline = {
        int(action["source_index"]): action.get("jianzi_text")
        for action in (item.get("baseline_plan", {}).get("actions") or [])
    }
    phrases = []
    for action in item.get("reference_plan", {}).get("actions") or []:
        text = str(action.get("jianzi_text") or action.get("text") or "").strip()
        if (len(text) >= 2 and canonical_jianzi_text(text)
                != canonical_jianzi_text(baseline.get(int(action["source_index"])))):
            phrases.append(text)
    return tuple(dict.fromkeys(phrases))





def text_blocks(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def response_content_blocks(response) -> list[dict[str, Any]]:
    """Serialize every response block for the next API turn, including thinking."""
    blocks = []
    for block in response.content:
        if hasattr(block, "model_dump"):
            blocks.append(block.model_dump(exclude_none=True))
        elif isinstance(block, dict):
            blocks.append(deepcopy(block))
        else:
            blocks.append({"type": getattr(block, "type", "unknown")})
    return blocks


def parse_final(text: str, *, required_keys: set[str] | None = None) -> dict:
    """Extract the final complete JSON object from a model response.

    Models sometimes wrap the object in Markdown fences or a short prose
    prefix/suffix.  ``raw_decode`` lets us stop at the end of each complete
    object instead of joining unrelated braces with ``rfind``.  We retain the
    last one: GLM can occasionally echo a malformed draft or a tool receipt
    before emitting its actual final envelope.
    When the object is truncated only by missing closing delimiters, a
    conservative delimiter repair is attempted; no tokens are changed.
    """
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    last_payload: dict | None = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = text[start:]
        try:
            payload, _ = decoder.raw_decode(candidate)
            if (isinstance(payload, dict)
                    and (required_keys is None or required_keys.issubset(payload))):
                last_payload = payload
                continue
        except json.JSONDecodeError as exc:
            last_error = exc

        # Only repair a syntactically balanced prefix with missing closing
        # delimiters.  A malformed middle token remains a hard failure.
        stack: list[str] = []
        in_string = False
        escaped = False
        pairs = {"}": "{", "]": "["}
        for token in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif token == "\\":
                    escaped = True
                elif token == '"':
                    in_string = False
                continue
            if token == '"':
                in_string = True
            elif token in "{[":
                stack.append(token)
            elif token in "}]":
                if not stack or stack[-1] != pairs[token]:
                    stack = []
                    break
                stack.pop()
        if not stack and not in_string:
            continue
        if in_string:
            candidate += '"'
        repaired = candidate + "".join(
            "}" if opener == "{" else "]" for opener in reversed(stack)
        )
        try:
            payload = json.loads(repaired)
            if (isinstance(payload, dict)
                    and (required_keys is None or required_keys.issubset(payload))):
                last_payload = payload
                continue
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_payload is not None:
        return last_payload
    if last_error is not None:
        raise last_error
    raise ValueError("final response has no JSON object")


def extract_decision_summary(payload: dict, *, forbidden_phrases: tuple[str, ...] = (),
                             allow_private_reasoning_leakage: bool = False) -> str:
    summary = str(payload.get("decision_summary") or "").strip()
    if not summary:
        raise ValueError("teacher response has no decision_summary")
    if not allow_private_reasoning_leakage:
        if any(token.lower() in summary.lower() for token in BANNED_TEXT):
            raise ValueError("decision_summary leaks teacher-private guidance")
        if any(phrase and phrase in summary for phrase in forbidden_phrases):
            raise ValueError("decision_summary repeats private reference text")
    return summary


def recover_prefixed_summary(text: str, payload: dict) -> dict:
    """Recover one unambiguous provider variant without accepting free prose.

    GLM occasionally emits public reasoning followed by a *valid* JSON object
    containing only ``tool_calls``.  The reasoning is still present and is
    unambiguously the prefix immediately before that one object; normalize it
    into the required envelope.  Do not attempt recovery for arbitrary JSON,
    missing calls, or an empty prefix.
    """
    if set(payload) != {"tool_calls"} or not isinstance(payload.get("tool_calls"), list):
        return payload
    start = text.find('{"tool_calls"')
    if start < 0:
        start = text.find('{\n  "tool_calls"')
    if start < 0:
        return payload
    summary = text[:start].strip()
    summary = re.sub(r"^【(?:公开思考|公开推理|公开 reasoning|公开理由)】\s*", "", summary)
    summary = re.sub(r"(?:【工具调用】|```json)\s*$", "", summary).strip()
    if not summary:
        return payload
    return {"decision_summary": summary, "tool_calls": payload["tool_calls"]}


def recover_labeled_teacher_envelope(text: str) -> dict | None:
    """Recover GLM's unambiguous Chinese-labelled envelope.

    A provider may emit ``decision_summary：...`` followed by
    ``tool_calls：[<valid JSON array>]``.  The call array is lossless JSON, but
    searching for the first ``{`` mistakes its first call object for the
    whole envelope.  Decode the labelled array directly and preserve the
    complete preceding summary.  Free prose without both labels is rejected.
    """
    # ``generate_one`` deliberately compacts whitespace before parsing, so a
    # provider may put the two labels on one line.  Anchor the envelope at
    # ``decision_summary`` and allow any whitespace before ``tool_calls``;
    # this remains unambiguous without accepting arbitrary free prose.
    match = re.match(
        r"^\s*decision_summary\s*[：:]\s*(.*)\s+tool_calls\s*[：:]\s*(\[)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    summary = match.group(1).strip()
    if not summary:
        return None
    try:
        calls, end = json.JSONDecoder().raw_decode(text[match.start(2):])
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list):
        return None
    trailing = text[match.start(2) + end:].strip()
    if trailing and trailing not in {"```"}:
        return None
    return {
        "decision_summary": summary,
        "tool_calls": calls,
    }


def recover_malformed_summary_envelope(text: str) -> dict | None:
    """Recover a valid call array when only the summary string is malformed.

    GLM occasionally puts an unescaped quote in ``decision_summary`` while
    the final ``tool_calls`` array itself remains valid JSON.  Rejecting that
    response discards an otherwise executable and fully structured tool turn.
    Recover only when both labelled fields are present and the call array can
    be decoded losslessly; the summary is retained as plain text rather than
    attempting to reinterpret or edit it as JSON.
    """
    matches = list(re.finditer(r'"tool_calls"\s*:\s*(\[)', text))
    if not matches:
        return None
    match = matches[-1]
    try:
        calls, _ = json.JSONDecoder().raw_decode(text[match.start(1):])
    except json.JSONDecodeError:
        # A recurrent GLM variant closes the arguments object but omits the
        # enclosing call-object brace (``...arguments:{...}]}]``).  The
        # labelled name plus independently decodable arguments object remain
        # lossless.  Extract only those explicit pairs; do not try to repair
        # arbitrary JSON tokens or fabricate missing argument values.
        calls = []
        call_pattern = re.compile(
            r'"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*'
            r'"arguments"\s*:\s*'
        )
        tail = text[match.start(1):]
        for call_match in call_pattern.finditer(tail):
            try:
                arguments, _ = json.JSONDecoder().raw_decode(
                    tail[call_match.end():]
                )
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                calls.append({"name": call_match.group("name"),
                              "arguments": arguments})
        if not calls:
            return None
    if not isinstance(calls, list):
        return None
    summary_matches = list(re.finditer(r'"decision_summary"\s*:\s*"', text[:match.start()]))
    if not summary_matches:
        return None
    summary = text[summary_matches[-1].end():match.start()].rstrip()
    if summary.endswith(","):
        summary = summary[:-1].rstrip()
    if summary.endswith('"'):
        summary = summary[:-1]
    if not summary.strip():
        return None
    return {"decision_summary": summary.strip(), "tool_calls": calls}


def recover_explicit_noop_prose(text: str) -> dict | None:
    """Normalize GLM's explicit no-op prose into the teacher envelope.

    GLM can follow the semantic no-op instruction while omitting the required
    JSON wrapper, e.g. saying that no rewrite is needed and ``edit_plan`` is
    not called.  Recover only this unambiguous variant; arbitrary prose must
    remain invalid so malformed tool responses are not silently accepted.
    """
    compact = text.strip()
    if not compact or "edit_plan" not in compact:
        return None
    if not re.search(r"无需(?:[^。；，,]{0,8})?(?:改写|润色|修改)|不需(?:[^。；，,]{0,8})?(?:改写|润色|修改)", compact):
        return None
    if not re.search(r"不调用\s*edit_plan|不使用\s*edit_plan|未调用\s*edit_plan", compact):
        return None
    return {"decision_summary": compact, "tool_calls": []}


def parse_teacher_envelope(text: str, *, forbidden_phrases: tuple[str, ...] = (),
                           allow_private_reasoning_leakage: bool = False) -> dict:
    """Parse the teacher's single JSON protocol for a batch of real tool calls."""
    payload = recover_labeled_teacher_envelope(text)
    if payload is None:
        try:
            payload = parse_final(text, required_keys={"tool_calls"})
        except ValueError as exc:
            payload = (recover_malformed_summary_envelope(text)
                       or recover_explicit_noop_prose(text))
            if payload is None:
                raise exc
    # Some providers wrap the requested envelope in a single ``tool_turn``
    # object.  Unwrap only this unambiguous shape; arbitrary extra fields are
    # still rejected below so malformed or contaminated responses do not pass.
    if (set(payload) == {"tool_turn"}
            and isinstance(payload.get("tool_turn"), dict)):
        payload = payload["tool_turn"]
    payload = recover_prefixed_summary(text, payload)
    if not isinstance(payload, dict):
        raise ValueError("teacher envelope must be a JSON object")
    # ``private_reasoning`` is a teacher-only scratchpad.  It may use the
    # private GQS/reference supplied to the teacher, so accepting it must not
    # make it part of the parsed envelope returned to the public trajectory
    # builder.  The raw provider response stays solely in teacher_io_trace
    # until the existing redaction step; public messages receive only the
    # validated decision_summary and tool calls below.
    allowed_fields = {"private_reasoning", "decision_summary", "tool_calls"}
    if set(payload) - allowed_fields or "tool_calls" not in payload:
        raise ValueError(
            "teacher envelope must contain tool_calls plus optional "
            "decision_summary/private_reasoning"
        )
    if "private_reasoning" in payload and not isinstance(payload["private_reasoning"], str):
        raise ValueError("private_reasoning must be a string when present")
    raw_summary = str(payload.get("decision_summary") or "").strip()
    if not raw_summary:
        raise ValueError("teacher envelope requires decision_summary before every tool call")
    summary = extract_decision_summary(
        payload,
        forbidden_phrases=forbidden_phrases,
        allow_private_reasoning_leakage=allow_private_reasoning_leakage,
    )
    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    # A small batch is useful for genuinely independent checks, but a model
    # must not turn one round into dozens of candidate queries.  It remains
    # free to choose a single-event or multi-event query shape.
    if len(calls) > 5:
        raise ValueError("teacher envelope permits at most five tool calls per assistant turn")
    normalized = []
    for position, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            raise ValueError(f"tool_calls[{position}] must be an object")
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments")
        # GLM occasionally flattens the sole edit_plan argument one level,
        # returning {"name": "edit_plan", "jianzi_rows": [...]} rather
        # than putting it under ``arguments``.  This is losslessly
        # equivalent to the documented protocol; normalize only this
        # unambiguous one-tool shape rather than accepting arbitrary fields.
        if (arguments is None and name == "edit_plan"
                and set(call) == {"name", "jianzi_rows"}):
            arguments = {"jianzi_rows": call["jianzi_rows"]}
        # Some providers serialize the arguments object one extra time.  A
        # valid JSON object string is losslessly equivalent and safe to
        # normalize; arbitrary text remains rejected below.
        if isinstance(arguments, str):
            try:
                decoded_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                decoded_arguments = None
            if isinstance(decoded_arguments, dict):
                arguments = decoded_arguments
        if not name or not isinstance(arguments, dict):
            raise ValueError(f"tool_calls[{position}] needs name and object arguments")
        normalized.append({"name": name, "arguments": arguments})
    return {"decision_summary": summary, "tool_calls": normalized}




















def generate_one(client, model: str, item: dict, stage: str, targets: list[dict],
                 historical: dict, max_rounds: int, *,
                 objective: str,
                 retry_notes: list[str] | None = None,
                 allow_private_reasoning_leakage: bool = False) -> tuple[dict, dict]:
    item = deepcopy(item)
    item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(
        item, historical
    )
    public_input = _public_input(item)
    direct_final = stage == "single_stage"
    basic = objective == "basic_fingering" or direct_final
    toward = objective == "toward_reference"
    review_noop = objective == "review_noop"
    if not (basic or toward or review_noop):
        raise ValueError(f"unsupported text-only objective: {objective}")
    public_system = public_system_for(stage, basic=basic)
    # Both stages may inspect the final annotation. Fingering is nevertheless
    # required to ground an independent basic draft in pitch candidates; the
    # second stage owns complex techniques and final display ranges.
    # A reasoned no-op review must be genuinely public-grounded.  The
    # selection of candidate phrases happens offline, but the teacher receives
    # neither the annotation text nor any reference-derived target patch.
    reference_gqs = None if review_noop else render_teacher_reference_gqs(item)
    private_instruction = {'role': public_system}
    private_instruction["tools"] = public_tools_for(stage, basic=basic)
    private_instruction["teacher_only_goal"] = {
        "objective": (
            "从空白谱面一次完成当前段可读、可演奏的最终减字、复杂技法和显示范围"
            if direct_final else
            ("为当前段每个演奏事件填写可读、可演奏的基础减字初稿"
             if basic else
            ("复核现有减字，并说明当前稿为何可以直接保留"
             if review_noop else
             "依据 GQS 与上下文润色最终减字、复杂技法和空显示范围"))
        )
    }
    private_instruction["rules"] = ([
        "逐音或按相邻音组分析现有取音、衔接、技法和显示范围为什么合理；只评价当前方案，不提出任何改写、补充、置空或省略建议。",
        f"直接给出可独立核查的音乐学复核意见，不调用工具；最后必须用“{NOOP_CONCLUSION}”收束。",
        "用“当前稿”或“当前谱面”指称现有内容，不讨论生成流程或数据来源。",
    ] if review_noop else [
        "私有标注可以用于内部决定哪里需要改、往什么方向改；decision_summary 会作为训练轨迹中的推理部分，用于训练模型推理出应该如何在没有参考标注时编辑完成专业的减字谱，因此不得提及 GQS、教师提示、系统提示、教师私有参考、最终标注、参考答案或目标答案等相关字眼，也不要出现不得写“与参考一致”“参考使用”或“按提示”等说法",
        "对新选或改写的按音、撮等双音/复合取声，在 decision_summary 的逐音或相邻音组分析中简短说明：左手为何选用该按指、该按位与另一按位能否同时落手；右手为何选该取声及其与目标弦的关系。理由以实际可演奏性、音高和前后衔接为主，避免空泛重复。",
        "**尽量要逐音说明为什么选择该左右手动作/手指；逐小段说明选择该减字在情感表达上的考虑。一定要详细说明原因，因为这涉及到学生模型能否真正学会复杂指法的意义**",
        "最终结束前每个演奏事件必须得到字符串；若前一减字已覆盖多个动作，后续行可用空字符串置空表示不重复显示。",
    ])
    # Keep the surface order explicit in the teacher-only prompt.  The model
    # otherwise sometimes copies the semantic field order (finger/string/hui)
    # into prose-like strings that are not conventional jianzipu notation.
    jianzi_field_order_rule = (
        "【减字字段次序】先在心中按固定模板组织，再写入 edit_plan：前置技法→取音状态（如泛音）"
        "→左手按指→徽位→右手指法→起音弦。成对例子：名指九徽勾六弦（不是名指六弦九徽勾）；"
        "泛音名指七徽勾四弦（不是泛音四弦七徽勾四弦）；"
        "名指九徽打三弦（不是名指三弦九徽打）；"
        "绰名指十徽八分勾三弦（不是绰名指三弦十徽八分勾）。"
        "除撮、泼、剌等明确的多弦复合写法外，一个起音的弦序只写在该音右手指法之后；"
        "完成前逐项比对这四组正例，修正任何把弦序写到徽位之前的行。"
    )
    if basic and not direct_final:
        private_instruction["rules"] = [
            jianzi_field_order_rule,
            "首次查询音高时，若当前段有4个或以上可解析的发音事件，尽量把至少4个（最好全部）音序放在同一次 get_pitch_candidates 的 event_indices 中；候选会按目标音高自动去重。只有后续确需核查某个单音时才逐音查询，不要把首次查询拆成逐音调用。",
            "参考 GQS 提供最终方向；不要逐行照抄其中的复杂走手、复合技法或空显示范围。",
            "Fingering 初稿只使用泛音、按音、散音和右手抹、挑、勾、剔、擘、托、打、摘、撮；绰上、注下、吟、猱、走手及其他复杂技法留给 Guqinizer。",
            "如果参考 GQS 含走手、复合技法或成片空显示，Fingering 应优先依据候选重建可读的基础取音初稿，而不是提交与参考 GQS 逐行完全相同的文本；这些最终化内容留给 Guqinizer。若参考 GQS 本身简单、基础且可直接演奏，则允许初稿与其一致，不要为了制造差异而改写。",
            "依据音高候选、简谱、ABC 与上下文，尽量为每个演奏事件填写基础取音减字；只有谱面关系明显需要留空时才使用空字符串。复杂润色留给 Guqinizer。",
            "可以先提交一部分已经确定的音；未提交的音不会使这次预览失败。继续根据工具结果分批编辑，直到结束前当前段所有发音事件都有减字或明确空字符串。",
            "基础取音阶段只允许泛音、按音、散音，右手只用抹、挑、勾、剔、擘、托、打、摘、撮；不要主动填写绰上、注下、吟、猱、走手或其他复杂技法。",
            "只查询有明确音高的起音；小节线不要提交。休止和延音不查音高，但若谱面需要表达走猱、猱、吟、泛止等延续动作，可以提交减字。",
        ] + private_reference_semantics_rules(stage) + private_instruction["rules"]
    elif not review_noop:
        private_instruction["rules"] = [
            jianzi_field_order_rule,
            "休止和延音行没有新音高，但若要表达走猱、猱、吟、泛止等延续动作，可以提交减字。",
            "给出的 GQS 参考是改进方向，不要求逐字复制；请结合基础初稿、上下文和专业判断适度加入高级指法。",
            "必须从头到尾按音序审阅完整个当前段：可以逐音分析，也可以每次按一小组相邻音分析，但不能只讨论少数差异音便结束。私有标注用于提示值得关注的位置和方向，不要求逐字照抄；公开 reasoning 应说明各音或各小组为何保留或修改，以及相关的音高、技法和减字取舍，不要说“因为标注/参考这样写”。",
            "遇到复杂技法时，应结合相邻音组说明其演奏前提和效果，例如承接的弦、按位、右手动作或余音状态；重点核对走手动作的同弦连续性，以及掐撮三声等组合技法此前是否已建立所需的撮弦、按散关系和按位。允许依据完整上下文作出与私有标注不同但合理的判断。",
            "如果逐音分析后确认当前基础初稿已经合适，不需要润色，则不要调用 edit_plan；tool_calls 返回空数组，并在 decision_summary 中逐音或按相邻音组说明为何现有指法、衔接和技法已经足够，无需为了制造变化而改写。",
        ] + private_reference_semantics_rules(stage) + private_instruction["rules"]
        if direct_final:
            private_instruction["rules"] = [
                jianzi_field_order_rule,
                "当前段从空白谱面开始：首次查询音高时，若有4个或以上可解析发音事件，尽量把至少4个（最好全部）音序放在同一次 get_pitch_candidates 的 event_indices 中；之后按候选、上下文与演奏可行性完成最终减字。",
                "不要先写一套受限的基础稿再等待第二阶段；本阶段直接决定需要的走手、复合技法、泛音区间和空显示，同时逐音检查音高与左右手可演奏性。",
            ] + private_instruction["rules"]
    # Retry diagnostics remain in the private failure trace only.  Do not add
    # them to the teacher prompt: they can distract the model and expose
    # implementation details unrelated to the musical decision.
    # 撮等双音的按指可达性在基础取音时就必须成立；不能只等
    # Guqinizer 再看到该条私有知识，否则错误的双按音已经进入中间稿。
    if basic or toward:
        private_instruction["rules"].extend(render_compound_gesture_knowledge(item))
    private_instruction["output_contract"] = ({
        "final_answer": f"直接输出最终复核意见文字，并以“{NOOP_CONCLUSION}”结尾；不要输出 JSON、tool_calls、Markdown 或代码围栏。",
    } if review_noop else {
        "tool_turn": {
            "private_reasoning": "可选；仅供教师内部思考，可引用私有 GQS/标注；"
                                 "系统会丢弃该字段，绝不进入训练轨迹或公开消息",
            "decision_summary": "必填；基于公开谱面、前文或已有工具结果的理由或思考过程"
                               "（不得含未转义的半角双引号或换行，引用参数用中文引号）",
            "tool_calls": [{"name": "工具名", "arguments": {"参数": "值"}}],
        },
        "rules": [
            "每轮只输出一个 JSON 对象，不要输出 Markdown、代码围栏或 JSON 之外的文字。",
            "private_reasoning 可选，仅用于你在私有参考与公开谱面之间作内部核对；"
            "可提及私有标注，但该字段会被丢弃，不能替代公开 decision_summary。",
            "每次工具调用前必须填写 decision_summary；把一个或多个调用写入 tool_calls。",
            "若分析后确定无需调用工具，也必须填写有音乐学内容的 decision_summary，并令 tool_calls=[]。",
            "工具 Schema 已在 tools 字段中给出；不得编造工具名或参数。",
            "每个 tool_calls 元素必须严格写成 {\"name\":\"工具名\",\"arguments\":{...}}；不得把参数直接放在元素内，不得使用 d/t/n/a 等缩写字段，也不得遗漏任何 JSON 括号。",
            "只有可见工具回执实际含有 :warning:音高不匹配 时，decision_summary 才能称其为音高警告；未出现该回执时只能描述候选核查或音乐判断。",
        ],
        "forbidden": (
            "decision_summary 不得提及 GQS、reference、teacher、target、教师提示、系统提示、提示词、教师私有参考、最终标注、标注答案或私有答案，"
            "不得把私有 GQS 中的具体减字内容包装成公开依据，"
            "不得复述隐藏推理过程。"
        ),
    })
    if stage in {"guqinization", "single_stage"}:
        item["public_pitch_warning_source_indices"] = sorted(
            public_pitch_warning_source_indices(item, historical=historical)
        )
    else:
        item.pop("public_pitch_warning_source_indices", None)
    user_payload = render_public_prompt(item, stage)
    api_messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_payload}
    ]
    public_messages: list[dict[str, Any]] = [
        {"role": "system", "content": public_system},
        {"role": "user", "content": user_payload},
    ]
    runtime = RealToolRuntime(
        item, historical, basic_fingering=basic,
        toward_reference=toward, target_patches=targets,
    )
    final_payload = None
    accepted_preview: list[dict] | None = None
    no_edit_accepted = False
    pitch_candidate_round: dict[int, int] = {}
    pitch_repair_attempted: set[int] = set()
    pending_pitch_warning_events: set[int] = set()
    submitted_edit_signatures: set[str] = set()
    teacher_io_trace: list[dict] = []
    for round_number in range(1, max_rounds + 1):
        options = {"model": model,
                   "max_tokens": (1600 if review_noop else
                                  int(os.getenv("GLM_MAX_TOKENS", "8000"))),
                   "temperature": 0.2,
                   "system": render_teacher_system(private_instruction, reference_gqs),
                   "messages": api_messages}
        # Native reasoning stays private; pass the full response content to
        # the next turn while exposing only text blocks publicly.
        thinking_mode = os.getenv("GLM_THINKING", "disabled").lower()
        options["thinking"] = (
            {"type": "adaptive"}
            if thinking_mode == "adaptive" else {"type": "disabled"}
        )
        response = client.messages.create(**options)
        teacher_io_trace.append({
            "round": round_number,
            "request": deepcopy(options),
            "response": response.model_dump(exclude_none=True),
        })
        raw_visible_text = re.sub(r"\s+", " ", text_blocks(response)).strip()
        if review_noop:
            visible_summary = raw_visible_text
            if not visible_summary:
                raise ValueError("reasoned no-op review returned an empty final answer")
            if len(visible_summary) > 3500:
                raise ValueError("reasoned no-op review is too long")
            if not re.search(r"无需修改|不需修改|直接保留|保留现状|保留当前|维持现状", visible_summary):
                raise ValueError("reasoned no-op review does not conclude that the draft should be retained")
            if re.search(
                r"edit_plan|调用工具|提交(?:以下)?修改|修改建议|建议将|建议修改|"
                r"(?:需要|应当|应该|仅需)\s*(?:补充|添加|改为|修改|调整)",
                visible_summary,
                flags=re.I,
            ):
                raise ValueError("reasoned no-op review proposes an edit")
            visible_summary = finalize_noop_review(visible_summary)
            teacher_calls = []
        else:
            envelope = parse_teacher_envelope(
                raw_visible_text,
                forbidden_phrases=private_only_reference_phrases(item),
                allow_private_reasoning_leakage=allow_private_reasoning_leakage,
            )
            visible_summary = envelope["decision_summary"]
            teacher_calls = envelope["tool_calls"]
        api_messages.append({
            "role": "assistant",
            "content": response_content_blocks(response),
        })
        results = []
        public_calls = []
        public_results = []
        for call_number, teacher_call in enumerate(teacher_calls, 1):
            name = teacher_call["name"]
            arguments = dict(teacher_call["arguments"])
            call_id = f"call_{round_number:02d}_{call_number:02d}"
            result = runtime.invoke(name, arguments)
            audit_call = runtime.calls[-1]
            audit_payload = ((audit_call.get("result") or {}).get("result") or {})
            if name == "get_pitch_candidates" and result.get("ok"):
                for query in audit_payload.get("queries") or []:
                    for event_index in query.get("event_indices") or []:
                        pitch_candidate_round[int(event_index)] = round_number
            call = {"id": call_id, "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(arguments, ensure_ascii=False)}}
            public_calls.append(call)
            public_results.append({"role": "tool", "tool_call_id": call_id,
                                   "name": name,
                                   "content": json.dumps(result, ensure_ascii=False)})
            results.append({"name": name, "arguments": arguments, "result": result})
            if (name == "edit_plan" and result.get("ok")
                    and result["result"].get("valid")):
                rows = arguments.get("jianzi_rows") or []
                signature = json.dumps(rows, ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":"))
                repeated_submission = bool(rows) and signature in submitted_edit_signatures
                if rows:
                    submitted_edit_signatures.add(signature)
                edited_events = {
                    int(row[0]) for row in arguments.get("jianzi_rows") or []
                    if isinstance(row, list) and row
                    and isinstance(row[0], int) and not isinstance(row[0], bool)
                }
                candidate_informed_edits = {
                    event_index for event_index in edited_events
                    if pitch_candidate_round.get(event_index, round_number) < round_number
                }
                current_warning_events = {
                    runtime._source_to_event(int(warning["source_index"]))
                    for warning in audit_payload.get("warnings") or []
                    if warning.get("code") == "jianzi_pitch_mismatch"
                    and warning.get("source_index") is not None
                }
                # A first edit may itself reveal a warning.  It cannot count
                # as its own repair: require another model turn so the warning
                # becomes visible in context before accepting the trajectory.
                unresolved_pitch_warning_events = advance_pitch_warning_state(
                    pending_pitch_warning_events,
                    pitch_repair_attempted,
                    current_warning_events,
                    candidate_informed_edits,
                )
                # 必须当前批次重放有效：畸形批次不得借助“空累计＝距离持平”
                # 的确认路径蒙混过关。
                accumulated = runtime.accumulated_patches
                if basic:
                    # A partial fingering edit is valid and is accumulated, but
                    # it is not an accepted final trajectory until all
                    # editable sounding events have a string value (including
                    # an explicit empty string where display should be blank).
                    preview_matches = validate_jianzi_only(
                        item, accumulated, toward_reference=False,
                        require_complete=True, historical=historical)[0]
                else:
                    preview_matches = validate_jianzi_only(
                        item, accumulated, toward_reference=False,
                        require_complete=False, historical=historical)[0]
                # A pitch warning is advisory.  It blocks acceptance only
                # until the teacher has inspected candidates and attempted a
                # later candidate-informed edit for that event.  Some
                # musically valid techniques cannot be represented by the
                # simple pitch parser, so requiring every warning to vanish
                # causes an unproductive correction loop.
                # A duplicate submission cannot provide new evidence or
                # change the score.  Preserve the accepted accumulated plan
                # rather than consuming more turns in an identical warning
                # loop.
                if repeated_submission and preview_matches:
                    accepted_preview = normalize_patches(accumulated)
                elif unresolved_pitch_warning_events:
                    accepted_preview = None
                elif preview_matches:
                    accepted_preview = normalize_patches(accumulated)
        public_assistant = {"role": "assistant", "content": visible_summary}
        if public_calls:
            public_assistant["tool_calls"] = public_calls
        if not teacher_calls:
            # An empty call list is a legitimate no-op only when the current
            # baseline is already complete (or the phrase has no editable
            # sounding events).  A blank fingering scaffold remains invalid,
            # so the model gets another turn instead of silently accepting it.
            if can_accept_empty_tool_turn(
                item, runtime.accumulated_patches, historical=historical
            ):
                public_messages.append(public_assistant)
                accepted_preview = normalize_patches(runtime.accumulated_patches)
                no_edit_accepted = not bool(runtime.accumulated_patches)
                final_payload = {"patches": accepted_preview}
                break
            # Keep the correction in both transcripts.  Previously it was
            # injected only into api_messages, so the teacher could refer to
            # a supposedly tool-reported missing-note warning that was absent
            # from the public training trajectory.  Also identify the actual
            # pending event indices instead of making the model guess.
            _, completeness = validate_jianzi_only(
                item, runtime.accumulated_patches, toward_reference=False,
                require_complete=True, historical=historical,
            )
            pending_source_indices = sorted({
                int(problem["source_index"])
                for problem in completeness.get("problems", [])
                if problem.get("code") == "pending_jianzi_text"
                and problem.get("source_index") is not None
            })
            pending_event_indices = [
                runtime._source_to_event(index) for index in pending_source_indices
            ]
            pending_text = (
                "；当前段仍待填写的音序："
                + "、".join(str(index) for index in pending_event_indices)
                if pending_event_indices else "；请检查当前段基础稿是否完整"
            )
            # This completeness feedback is an internal control message, not
            # a real edit_plan response; keep it out of public_messages.
            correction = {
                "tool_results": [{
                    "ok": False,
                    "error": (
                        "当前段仍有待填写的演奏音"
                        + pending_text
                        + "；不能以空工具调用结束，请通过 edit_plan.jianzi_rows 填写这些行。"
                    ),
                }],
                "instruction": "根据这条校验反馈继续；需要编辑时调用 edit_plan。",
            }
            api_messages.append({
                "role": "user",
                "content": json.dumps(correction, ensure_ascii=False),
            })
            continue
        public_messages.append(public_assistant)
        public_messages.extend(public_results)
        if accepted_preview is not None:
            final_payload = {"patches": accepted_preview}
            public_messages.append({
                "role": "assistant",
                "content": (
                    "工具预览已通过，当前段基础减字填写完成。"
                    if basic and not direct_final else "工具预览已通过，当前段最终减字填写完成。"
                ),
            })
            break
        unresolved_pitch_warning_events = (
            pending_pitch_warning_events - pitch_repair_attempted
        )
        if unresolved_pitch_warning_events:
            warning_indices = sorted(unresolved_pitch_warning_events)
            queried = all(
                index in pitch_candidate_round
                for index in warning_indices
            )
            if queried:
                instruction = (
                    "编辑工具报出音高不匹配。你已经查看过这些音的候选；"
                    f"请判断音序 {warning_indices} 是否需要修正。需要时只提交确实变化的行；"
                    "若基于候选与音乐判断决定保留，可用 tool_calls=[] 结束并说明理由。"
                )
            else:
                instruction = (
                    "编辑工具报出音高不匹配。先查看这些音的候选，再决定是否修正："
                    f"音序 {warning_indices}；下一轮调用 get_pitch_candidates，"
                    "把这些音序放入 event_indices。看到真实候选后，可修正或保留并结束。"
                )
        else:
            instruction = "根据这些真实工具结果继续；需要工具时输出 tool_calls。"
        api_messages.append({"role": "user", "content": json.dumps({
            "tool_results": results,
            "instruction": instruction,
        }, ensure_ascii=False)})
    if final_payload is None:
        raise ValueError("teacher exceeded tool-round limit")
    proposed = normalize_patches(final_payload.get("patches") or [])
    if not isinstance(proposed, list):
        raise ValueError("final JSON has no patches list")
    replay = replay_patches(item["baseline_plan"], proposed, strict_before=not toward)
    problems = compare_replay_to_patch_targets(replay.actions, proposed)
    if not replay.valid or problems:
        raise ValueError(f"final patches do not replay: {replay.errors + problems}")
    if toward:
        valid_toward, toward_report = validate_jianzi_only(
            item, proposed, toward_reference=True, historical=historical)
        if not valid_toward:
            raise ValueError(f"Guqinizer phrase quality audit failed: {toward_report}")
    if basic:
        valid_basic, basic_report = validate_jianzi_only(
            item, proposed, toward_reference=False, historical=historical)
        if not valid_basic:
            raise ValueError(f"basic fingering audit failed: {basic_report['problems']}")
    if not no_edit_accepted and not any(
            call["name"] == "edit_plan" and call["result"].get("ok")
            for call in runtime.calls):
        raise ValueError("no successful edit_plan preview")
    serialized = json.dumps(public_messages, ensure_ascii=False)
    if any(f'"{key}"' in serialized for key in
           ("teacher_private", "reference_plan", "reference_actions", "target_patches")):
        raise ValueError("private key leaked into public messages")
    sample_id = f"{item['trajectory_id']}-{stage}-teacher-tools"
    public = {"schema_version": "agent-messages-2.1", "sample_id": sample_id,
              "agent_stage": stage, "generation_mode": "teacher_autonomous_real_tools",
              "tools": public_tools_for(stage, basic=basic),
              "messages": public_messages,
              "provenance": {"source_trajectory_id": item["trajectory_id"],
                             "score_key": item["score_key"], "split": item["split"]},
              "termination": {"kind": "no_changes" if no_edit_accepted
                               else "accepted_edit_plan"},
              "verification": {"tools_really_executed": True, "replay_passed": True,
                               "private_leakage_passed": True}}
    final_report = validate_jianzi_only(
        item, proposed, toward_reference=toward, historical=historical
    )[1]
    accepted_plan = final_report["plan"]
    public["verification"]["training_eligible"] = bool(
        final_report.get("offline_quality", {}).get("training_eligible", False))
    private = {"sample_id": sample_id, "teacher_private": {
        "target_patches": targets, "tool_execution_log": runtime.calls,
        "teacher_io_trace": teacher_io_trace,
        "teacher_model": model, "objective": objective,
        "accepted_patches": proposed,
        "accepted_plan": accepted_plan,
        "jianzi_quality_report": final_report,
        "annotation_gqs": render_annotation_gqs(item)},
        "verification": public["verification"]}
    return public, private


class _RecordingMessages:
    def __init__(self, messages, trace: list[dict]):
        self._messages = messages
        self._trace = trace

    def create(self, **options):
        entry = {"request": deepcopy(options)}
        try:
            response = self._messages.create(**options)
            entry["response"] = response.model_dump(exclude_none=True)
            return response
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._trace.append(entry)


class _RecordingClient:
    def __init__(self, client, trace: list[dict]):
        self.messages = _RecordingMessages(client.messages, trace)


class _RateLimitedMessages:
    """Rate-limit local calls and, optionally, all worker processes together."""

    def __init__(self, messages, min_interval: float):
        self._messages = messages
        self._min_interval = max(0.0, float(min_interval))
        self._last_call = 0.0
        self._global_min_interval = max(
            0.0, float(os.getenv("GLM_GLOBAL_MIN_INTERVAL", "0"))
        )
        state = os.getenv("GLM_GLOBAL_RATE_LIMIT_STATE", "").strip()
        self._global_state_path = Path(state) if state else None

    def _reserve_global_slot(self) -> None:
        """Reserve one cross-process request slot when configured.

        ``--min-interval`` only spaces calls made by one worker.  The batch
        runner uses separate processes, so a small shared lock file is needed
        to prevent simultaneous retries from creating a 429 burst.
        """
        if not self._global_state_path or self._global_min_interval <= 0:
            return
        self._global_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._global_state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                try:
                    previous = float(handle.read().strip() or "0")
                except ValueError:
                    previous = 0.0
                now = time.monotonic()
                # ``monotonic`` restarts at boot.  A persisted timestamp from
                # a prior macOS/Linux boot must never turn into a days-long
                # artificial wait for every resumed worker.
                if previous > now:
                    previous = 0.0
                wait = self._global_min_interval - (now - previous)
                if wait > 0:
                    time.sleep(wait)
                handle.seek(0)
                handle.truncate()
                handle.write(f"{time.monotonic():.9f}")
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def create(self, **options):
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._reserve_global_slot()
        self._last_call = time.monotonic()
        return self._messages.create(**options)


class _RateLimitedClient:
    def __init__(self, client, min_interval: float):
        self.messages = _RateLimitedMessages(client.messages, min_interval)


def generate_with_retries(client, model: str, item: dict, stage: str,
                          targets: list[dict], historical: dict, max_rounds: int,
                          attempts: int, *, objective: str,
                          failure_traces: list[dict] | None = None,
                          allow_private_reasoning_leakage: bool = False):
    errors = []
    retry_notes: list[str] = []
    for attempt_number in range(1, attempts + 1):
        trace: list[dict] = []
        try:
            return generate_one(_RecordingClient(client, trace), model, item, stage,
                                targets, historical,
                                max_rounds, objective=objective,
                                retry_notes=retry_notes,
                                allow_private_reasoning_leakage=allow_private_reasoning_leakage)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            retry_notes = [error[:300]]
            if failure_traces is not None:
                failure_traces.append({
                    "trajectory_id": item["trajectory_id"], "stage": stage,
                    "attempt": attempt_number, "model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "teacher_io_trace": trace,
                })
            transient = (
                "RateLimitError" in error or "rate_limit_error" in error
                or "429" in error or "APIConnectionError" in error
                or "ConnectError" in error or "ReadError" in error
                or "TimeoutError" in error or "timed out" in error.lower()
            )
            if transient and attempt_number < attempts:
                delay = min(60.0, 5.0 * (2 ** (attempt_number - 1))) + random.uniform(0.0, 2.0)
                print(f"transient API error: retrying {item['trajectory_id']} {stage} "
                      f"after {delay:.1f}s; cause={error}", flush=True)
                time.sleep(delay)
    raise RuntimeError("teacher attempts exhausted: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ABC_J/agent_training/inferred_v6/inferred_trajectories_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages_teacher_tools")
    parser.add_argument("--stage", choices=("fingering_agent", "guqinization"),
                        help="limit generation to one agent stage")
    parser.add_argument("--max-actions", type=int,
                        help="smoke-test filter by Route Search action count")
    parser.add_argument("--trajectory-id", action="append",
                        help="limit to one or more exact inferred trajectory IDs")
    parser.add_argument("--trajectory-id-file", type=Path,
                        help="newline-delimited trajectory IDs; avoids very long command lines")
    parser.add_argument("--trajectory-id-report", type=Path,
                        help="JSON report containing a no_op_ids/retry_ids style trajectory-ID list")
    parser.add_argument("--trajectory-id-report-key", default="no_op_ids",
                        help="list-valued key read from --trajectory-id-report (default: no_op_ids)")
    parser.add_argument("--intermediate-input", type=Path,
                        help="resume Guqinizer from previously accepted fingering_intermediates.jsonl")
    parser.add_argument("--max-tool-rounds", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-interval", type=float,
                        help="minimum seconds between serialized API requests (default: TEACHER_MIN_INTERVAL or 2)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing output directory without duplicating completed trajectory IDs")
    parser.add_argument("--retry-failed", action="store_true",
                        help="with --resume, retry only stages previously recorded as attempted_with_failure")
    parser.add_argument("--allow-private-reasoning-leakage", action="store_true",
                        help="accept raw decision_summary that mentions private references; intended before redaction")
    parser.add_argument("--include-guqinizer-no-op", action="store_true",
                        help="run Guqinizer even when text comparison infers no edit target, so it can produce a reasoned no-op trajectory")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split the selected pool across N parallel worker processes")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="this worker's position in 0..shard-count-1")
    parser.add_argument("--score-shard-count", type=int, default=1,
                        help="stream only one deterministic score shard, reducing memory for parallel workers")
    parser.add_argument("--score-shard-index", type=int, default=0,
                        help="score shard position in 0..score-shard-count-1")
    parser.add_argument("--model")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    from anthropic import Anthropic
    model = args.model or os.getenv("GLM_MODEL", "glm-5.3")
    if not model.lower().startswith("glm"):
        raise RuntimeError("This teacher pipeline is GLM-only; refusing to call a non-GLM model")
    api_key = os.getenv("GLM_API_KEY")
    base_url = os.getenv("GLM_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("GLM API credentials are not configured")
    min_interval = (args.min_interval if args.min_interval is not None else
                    float(os.getenv("TEACHER_MIN_INTERVAL", "2")))
    client = Anthropic(api_key=api_key,
                       base_url=base_url.rstrip("/"),
                       timeout=120.0, max_retries=1)
    client = _RateLimitedClient(client, min_interval)
    if args.score_shard_count < 1 or not 0 <= args.score_shard_index < args.score_shard_count:
        raise SystemExit("--score-shard-index must be in 0..score-shard-count-1")
    def score_bucket(score_key: str) -> int:
        # Stable across processes (unlike Python's randomized hash), while
        # keeping every score's full phrase history in one worker.
        digest = hashlib.blake2b(str(score_key).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % args.score_shard_count
    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            candidate = json.loads(line)
            validate_normalized_tuning(candidate)
            if args.score_shard_count > 1 and score_bucket(candidate.get("score_key", "")) != args.score_shard_index:
                continue
            rows.append(candidate)
    historical = {(row["score_key"], row["phrase_id"]): row for row in rows}
    resumed = {}
    if args.intermediate_input:
        resumed = {row["trajectory_id"]: row for row in (
            json.loads(line) for line in args.intermediate_input.open(encoding="utf-8"))}
    # A phrase whose annotation is wholly blank cannot supervise either
    # stage.  Exclude it before selection/API calls rather than manufacturing
    # a Base draft and then teaching Guqinizer to erase the entire phrase.
    eligible = [row for row in rows if not is_all_empty_reference_phrase(row)]
    if args.trajectory_id or args.trajectory_id_file or args.trajectory_id_report:
        wanted = set(args.trajectory_id or [])
        if args.trajectory_id_file:
            wanted.update(line.strip() for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines()
                          if line.strip())
        if args.trajectory_id_report:
            report_payload = json.loads(args.trajectory_id_report.read_text(encoding="utf-8"))
            report_ids = report_payload.get(args.trajectory_id_report_key)
            if not isinstance(report_ids, list) or not all(isinstance(value, str) for value in report_ids):
                raise SystemExit(
                    f"--trajectory-id-report-key {args.trajectory_id_report_key!r} must name a string list"
                )
            wanted.update(report_ids)
        eligible = [row for row in eligible if row["trajectory_id"] in wanted]
    if resumed:
        eligible = [row for row in eligible if row["trajectory_id"] in resumed]
    if args.max_actions is not None:
        eligible = [row for row in eligible
                    if len(row["baseline_plan"].get("actions", [])) <= args.max_actions]
    if not 0 <= args.shard_index < max(args.shard_count, 1):
        raise SystemExit("--shard-index must be in 0..shard-count-1")
    if args.shard_count > 1:
        # Interleave so shards stay balanced across scores and phrase lengths.
        eligible = [row for position, row in enumerate(eligible)
                    if position % args.shard_count == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "public": args.output_dir / "messages_train.jsonl",
        "private": args.output_dir / "teacher_trajectory_audit.jsonl",
        "intermediates": args.output_dir / "fingering_intermediates.jsonl",
        "rejected": args.output_dir / "teacher_rejected_io.jsonl",
        "checkpoint": args.output_dir / "checkpoint.jsonl",
    }
    if not args.resume and any(path.exists() for path in output_paths.values()):
        raise SystemExit(f"output directory is not empty; use --resume: {args.output_dir}")

    def read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    existing_public = read_jsonl(output_paths["public"])
    existing_private = read_jsonl(output_paths["private"])
    existing_intermediates = read_jsonl(output_paths["intermediates"])
    existing_rejected = read_jsonl(output_paths["rejected"])
    existing_checkpoint = read_jsonl(output_paths["checkpoint"])
    existing_report = (json.loads((args.output_dir / "generation_report.json").read_text(encoding="utf-8"))
                       if args.resume and (args.output_dir / "generation_report.json").exists() else {})
    resumed.update({row["trajectory_id"]: row for row in existing_intermediates})
    stage_ids: dict[str, set[str]] = {}
    for row in existing_private + existing_public:
        match = re.match(r"^(.*)-(fingering_agent|guqinization)-teacher-tools$",
                         str(row.get("sample_id") or ""))
        if match:
            stage_ids.setdefault(match.group(1), set()).add(match.group(2))
    # Rejected traces are not completed stages.  They are treated as skipped
    # on an ordinary resume via the checkpoint, but remain eligible for an
    # explicit --retry-failed pass.
    checkpoint_statuses = {"completed"}
    if not args.retry_failed:
        checkpoint_statuses.add("attempted_with_failure")
    completed_ids = {str(row.get("trajectory_id")) for row in existing_checkpoint
                     if row.get("status") in checkpoint_statuses}
    resolved_stages: dict[str, set[str]] = {
        source_id: set(stages) for source_id, stages in stage_ids.items()
    }
    completed_ids.update(source_id for source_id, stages in stage_ids.items()
                         if "guqinization" in stages)
    # Ordinarily a phrase with no inferred Guqinizer target is complete after
    # Fingering.  An explicit no-op generation pass deliberately keeps it in
    # the pool to teach the model how to analyse and stop without editing.
    if not args.include_guqinizer_no_op:
        for source_id, intermediate in resumed.items():
            item = next((row for row in rows if row["trajectory_id"] == source_id), None)
            if item is None:
                continue
            references = item["reference_plan"]["actions"]
            if not infer_jianzi_text_patches(intermediate["plan"], references)["patches"]:
                completed_ids.add(source_id)
    pool = [row for row in eligible if row["trajectory_id"] not in completed_ids]
    # The already-filtered eligible pool is the complete requested workload.
    # In particular, an explicit --trajectory-id set must never be silently
    # truncated by an unrelated default limit.
    selected = pool
    counts = Counter(row.get("agent_stage") for row in existing_public)
    failures = list(existing_report.get("failures") or [])
    quarantined = list(existing_report.get("quarantine") or [])
    rejected_traces: list[dict] = []
    file_mode = "a" if args.resume else "w"
    checkpoint_mode = "a" if args.resume else "w"
    with (output_paths["public"].open(file_mode, encoding="utf-8", newline="\n") as pub,
          output_paths["private"].open(file_mode, encoding="utf-8", newline="\n") as priv,
          output_paths["intermediates"].open(file_mode, encoding="utf-8", newline="\n") as intermediates,
          output_paths["checkpoint"].open(checkpoint_mode, encoding="utf-8", newline="\n") as checkpoint):
        progress = tqdm(
            enumerate(selected, 1),
            total=len(selected),
            desc=f"{args.stage} trajectories",
            unit="phrase",
            dynamic_ncols=False,
            file=sys.stdout,
        )
        for item_number, item in progress:
            progress.set_postfix_str(str(item["trajectory_id"]), refresh=False)
            print(f"phrase {item_number}/{len(selected)}: {item['trajectory_id']}", flush=True)
            def flush_outputs() -> None:
                pub.flush()
                priv.flush()
                intermediates.flush()

            stage_item = deepcopy(item)
            if item["trajectory_id"] in resumed:
                intermediate_record = resumed[item["trajectory_id"]]
                intermediate_plan = intermediate_record["plan"]
                stage_item["baseline_plan"] = intermediate_plan
                flush_outputs()
            else:
                stage_item["baseline_plan"] = blank_plan_from_item(stage_item)
                try:
                    public, private = generate_with_retries(
                        client, model, stage_item, "fingering_agent", [], historical,
                        args.max_tool_rounds, args.max_attempts,
                        objective="basic_fingering", failure_traces=rejected_traces,
                        allow_private_reasoning_leakage=args.allow_private_reasoning_leakage)
                    priv.write(json.dumps(private, ensure_ascii=False) + "\n")
                    if public["verification"].get("training_eligible"):
                        pub.write(json.dumps(public, ensure_ascii=False) + "\n")
                        counts["fingering_agent"] += 1
                    else:
                        quarantined.append({"trajectory_id": item["trajectory_id"],
                                            "stage": "fingering_agent",
                                            "reason": "offline_jianzi_pitch_filter"})
                    intermediate_plan = private["teacher_private"]["accepted_plan"]
                    stage_item["baseline_plan"] = intermediate_plan
                    intermediates.write(json.dumps({
                        "trajectory_id": item["trajectory_id"],
                        "score_key": item["score_key"], "phrase_id": item["phrase_id"],
                        "normalized_tuning": item["input"]["normalized_tuning"],
                        "plan": intermediate_plan,
                        "teacher_model": model,
                        "verification": public["verification"],
                    }, ensure_ascii=False) + "\n")
                    resolved_stages.setdefault(item["trajectory_id"], set()).add("fingering_agent")
                    flush_outputs()
                except Exception as exc:
                    failures.append({"trajectory_id": item["trajectory_id"],
                                     "stage": "fingering_agent",
                                     "error": f"{type(exc).__name__}: {exc}"})
                    checkpoint.write(json.dumps({"trajectory_id": item["trajectory_id"],
                                                 "status": "attempted_with_failure",
                                                 "stage": "fingering_agent"}, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    flush_outputs()
                    continue
            references = item["reference_plan"]["actions"]
            inferred = infer_jianzi_text_patches(intermediate_plan, references)
            guqin_targets = inferred["patches"]
            run_guqinizer = bool(guqin_targets) or (
                args.include_guqinizer_no_op and args.stage != "fingering_agent"
            )
            guqin_done = not run_guqinizer or args.stage == "fingering_agent"
            if run_guqinizer and args.stage != "fingering_agent":
                try:
                    public, private = generate_with_retries(
                        client, model, stage_item, "guqinization", guqin_targets,
                        historical, args.max_tool_rounds, args.max_attempts,
                        objective=("review_noop" if not guqin_targets else "toward_reference"),
                        failure_traces=rejected_traces,
                        allow_private_reasoning_leakage=args.allow_private_reasoning_leakage)
                    priv.write(json.dumps(private, ensure_ascii=False) + "\n")
                    if public["verification"].get("training_eligible"):
                        pub.write(json.dumps(public, ensure_ascii=False) + "\n")
                        counts["guqinization"] += 1
                    else:
                        quarantined.append({"trajectory_id": item["trajectory_id"],
                                            "stage": "guqinization",
                                            "reason": "offline_jianzi_pitch_filter"})
                    guqin_done = True
                    resolved_stages.setdefault(item["trajectory_id"], set()).add("guqinization")
                    flush_outputs()
                except Exception as exc:
                    failures.append({"trajectory_id": item["trajectory_id"],
                                     "stage": "guqinization",
                                     "error": f"{type(exc).__name__}: {exc}"})
                    flush_outputs()
            if item["trajectory_id"] not in completed_ids and guqin_done:
                checkpoint.write(json.dumps({"trajectory_id": item["trajectory_id"],
                                             "status": "completed"}, ensure_ascii=False) + "\n")
                checkpoint.flush()
                completed_ids.add(item["trajectory_id"])
            elif item["trajectory_id"] not in completed_ids:
                checkpoint.write(json.dumps({"trajectory_id": item["trajectory_id"],
                                             "status": "attempted_with_failure"}, ensure_ascii=False) + "\n")
                checkpoint.flush()
            print(f"completed {item['trajectory_id']}: {dict(counts)}", flush=True)
            continue
    attempted_ids = (set(completed_ids)
                     | {row["trajectory_id"] for row in selected}
                     | {str(row.get("trajectory_id")) for row in failures
                        if row.get("trajectory_id")})
    if args.retry_failed:
        failures = [failure for failure in failures
                    if str(failure.get("trajectory_id")) not in completed_ids
                    and failure.get("stage") not in resolved_stages.get(
                        str(failure.get("trajectory_id")), set())]
    # A retry appends to the historical report.  Keep the latest unresolved
    # result per trajectory/stage so ``rejected`` reflects actual work still
    # pending rather than accumulating duplicate failure records.
    latest_failures: dict[tuple[str, str], dict] = {}
    for failure in failures:
        key = (str(failure.get("trajectory_id") or ""),
               str(failure.get("stage") or ""))
        latest_failures[key] = failure
    failures = list(latest_failures.values())
    report = {"schema_version": "teacher-real-tools-report-1.0", "model": model,
              "workflow": "basic_intermediate_to_annotation",
              "raw_reasoning_leakage_allowed": bool(args.allow_private_reasoning_leakage),
              "retry_failed": bool(args.retry_failed),
              "source_phrases": len(attempted_ids), "accepted": sum(counts.values()),
              "by_stage": dict(counts), "quarantined": len(quarantined),
              "quarantine": quarantined,
              "rejected": len(failures), "failures": failures}
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_paths["rejected"].open("a" if args.resume else "w",
                                       encoding="utf-8", newline="\n") as rejected_handle:
        for rejected in rejected_traces:
            rejected_handle.write(json.dumps(rejected, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if counts else 1


if __name__ == "__main__":
    raise SystemExit(main())
