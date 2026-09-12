#!/usr/bin/env python3
"""Generate answer-guided trajectories whose observable tools are really executed."""
from __future__ import annotations

import argparse
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
    {"name": "list_context", "description": "仅列出当前 phrase 之前可展开的 phrase 目录，不返回谱面动作。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "expand_context", "description": "展开一个未在 user 中提供的更早 phrase 的只读已确认谱面。已直接给出的“只读前一段”不得重复调用。返回与当前谱面相同的简表。",
     "input_schema": {"type": "object", "properties": {
         "phrase_id": {"type": "string"}}, "required": ["phrase_id"]}},
    {"name": "get_pitch_candidates", "description": "查询取音位置候选。参数使用当前段连续的音序（小节线不占音序）；首次查询时，若当前段有4个或以上可解析的发音事件，尽量把至少4个（最好全部）音序一次放入 event_indices；后续只有确需聚焦核查某个音时才单独查询。event_indices 中偶然包含小节线、休止或无法解析音高的序号时会跳过这些序号，继续处理其余有效音，并在结果中说明。类型可传一个字符串或字符串数组。工具会解析调号、八度与和弦，并按实际目标 MIDI 自动去重，每个音高只返回一份候选及其全部来源音序。target_midi 仅用于脱离谱面事件的特殊查询。",
     "input_schema": {"type": "object", "properties": {
         "event_index": {"type": "integer"},
         "event_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
         "target_midi": {"type": "number"},
         "类型": {"oneOf": [{"type": "string", "enum": ["泛音", "散音", "按音"]}, {"type": "array", "items": {"type": "string", "enum": ["泛音", "散音", "按音"]}, "minItems": 1}]},
         "max_candidates": {"type": "integer"}},
      "anyOf": [{"required": ["event_index"]}, {"required": ["event_indices"]},
                {"required": ["target_midi"]}]}},
    {"name": "edit_plan", "description": "批量预览并设置减字文本。唯一输入为 jianzi_rows=[[音序,减字文字],...]；音序应为 JSON 整数（例如 [12,\"吟\"]，不要写成 [\"12\",\"吟\"]），小节线没有音序、不可提交。一次调用可以只提交已决定修改的部分行，未提交的行不会使本次预览失败；Fingering 阶段应继续分批填写，直到结束前所有发音事件都有字符串（必要时为空字符串）。空字符串的直接语义是将该行 jianzi_text 置空；不会删除声音或其他演奏状态。若前一减字已包含多个动作，后续被覆盖的声音可用空字符串表示不重复显示。工具只在减字和简谱都可高置信解析且音高明显不符时给出警告。",
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
    if basic or stage == "guqinization":
        allowed.add("get_pitch_candidates")
    return [deepcopy(tool) for tool in TOOLS if tool["name"] in allowed]


def public_system_for(stage: str, *, basic: bool = False) -> str:
    prompt = PUBLIC_SYSTEM[stage] + (PURE_CUO_RULE if basic else "") + INHERITANCE_RULES_PROMPT + (
        " 每次工具调用前说明理由。"
        "edit_plan 一次可提交多行，也可以只提交当前已经确定的部分行；未提交的行保留现状，不会因尚未决定而使本次预览失败。收到反馈后只修正需要变化的行，不要重复未变化内容。"
    )
    if stage == "guqinization":
        prompt += (
            "先逐一分析每个音应该使用什么指法/减字，再决定是否编辑；"
            "只把决定改写的音提交给 edit_plan。"
        )
    if basic:
        prompt += (
            " 首次查询音高时，若当前段有4个或以上可解析的发音事件，尽量把至少4个（最好全部）音序放在同一次 get_pitch_candidates 的 event_indices 中；工具会按实际音高自动去重。只有后续确需核查某个单音时才逐音查询，不要把首次查询拆成逐音调用。"
            "依据候选位置、上下文和专业判断写出基础减字。尽量把每个演奏事件的减字填写完整，除非谱面关系很明显需要留空。"
            "可以分批提交已经确定的音，继续编辑直到结束前所有发音事件都有减字或明确空字符串。"
            "小节线不要提交；休止和延音没有新音高，但若要表达走猱、猱、吟、泛止等延续动作，可以提交对应减字。"
        )
    return prompt


def target_midis(item: dict, source_index: int) -> list[float]:
    """Every sounding target of a note: all ABC chord tones, else jianpu.

    Chord tokens are authoritative for 撮-style double stops; the jianpu field
    only carries one of the tones, so it is used solely as fallback.
    """
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    tonic = AUDIT.parse_tonic_midi(item["input"]["metadata"])
    # 和弦（撮等）的音高以存储双谱字为准：ABC 转换在降号调上存在半音偏差，
    # 记谱本身才是真相；无双谱字时回退到 ABC 和弦。
    if abc_chord_tones(str(note.get("abc") or "")):
        values = [AUDIT.parse_jianpu(note.get(field), tonic)
                  for field in ("jianpu", "jianpu_alt")]
        values = [float(value) for value in values if value is not None]
        if len(values) == 2:
            return values
        midis = abc_chord_midis(str(note.get("abc") or ""))
        if midis:
            return midis
    values = [AUDIT.parse_jianpu(note.get(field), tonic) for field in ("jianpu", "jianpu_alt")]
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError(f"source_index {source_index} has no sounding target")
    return [float(value) for value in values]


def render_pitch_candidate_table(target: float, candidates: list[dict],
                                 modes: tuple[str, ...]) -> str:
    mode_names = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
    lines = [
        f"指定音高｜MIDI {target:g}｜范围｜{'、'.join(mode_names.get(m, m) for m in modes)}",
        "序号｜方式｜弦徽｜实得MIDI｜可信度",
    ]
    for rank, candidate in enumerate(candidates, 1):
        mode = mode_names.get(candidate.get("mode"), str(candidate.get("mode") or "未知"))
        position = f'{candidate.get("string")}弦'
        if candidate.get("mode") not in {"open"} and candidate.get("hui") is not None:
            position += hui_label(candidate.get("hui"))
        confidence = "精确" if candidate.get("confidence") == "exact" else "近似"
        lines.append(
            f'{rank}｜{mode}｜{position}｜{float(candidate.get("sounding_midi")):.1f}｜{confidence}'
        )
    if not candidates:
        lines.append("无符合容差的候选")
    return "\n".join(lines)


def pitch_label(item: dict, source_index: int) -> str:
    """Jianpu label of a note; chord notes show both stored digits."""
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    written = note.get("jianpu") or note.get("jianpu_alt") or "休止"
    return chord_jianpu_label(note.get("abc"), written, note.get("jianpu_alt"))


def event_index_for_source(item: dict, source_index: int) -> int | None:
    """Translate an internal sparse source index to the visible continuous 音序."""
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        if int(note.get("index")) == int(source_index):
            value = note.get("event_index")
            return int(value) if value is not None else None
    return None


def source_index_for_event(item: dict, event_index: int) -> int | None:
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        value = note.get("event_index")
        if value is not None and int(value) == int(event_index):
            return int(note["index"])
    return None


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


def render_grouped_candidate_table(target: float, candidates: list[dict],
                                   modes: tuple[str, ...],
                                   sources: list[dict]) -> str:
    """Render one physical candidate list with every score event that reuses it."""
    source_text = "、".join(
        str(source.get("event_index", source["source_index"])) for source in sources
    )
    jianpu_values = list(dict.fromkeys(
        str(source.get("jianpu") or "") for source in sources
        if str(source.get("jianpu") or "")
    ))
    lines = render_pitch_candidate_table(target, candidates, modes).splitlines()
    lines[0] = f"目标音高｜MIDI {target:g}｜来源｜{source_text}"
    if jianpu_values:
        lines[0] += f"｜简谱｜{'、'.join(jianpu_values)}"
    return "\n".join(lines)


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
    warning_rule = (
        '工具在已填写的减字后标注":warning:音高不匹配"时，应尽量核对该行的取音与减字并修正；'
        '这是非阻塞警告，处理该行后继续完成其他合法编辑。'
    )
    if stage == "fingering_agent":
        return [
            f'私有参考表中的"{OMITTED_PLACEHOLDER}"表示再作谱面省略，不是空标注，也不是未知值；该音仍有演奏语义，不得仅因该标记删除声音；该标记本身也是允许提交的减字文字，可通过 edit_plan.jianzi_rows 直接填写"{OMITTED_PLACEHOLDER}"；这表示沿用再作动作继承的减字（前一段或标记处），不等于把该音置为空字符串。',
            warning_rule,
        ]
    return [
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


def harmonic_region_at_phrase_start(
    item: dict, historical: dict[tuple[str, str], dict]
) -> bool:
    """Replay confirmed 泛起/泛止 markers before the current phrase."""
    current_start = int(item["input"]["event_range"]["start"])
    earlier = sorted(
        (row for (score_key, _), row in historical.items()
         if score_key == item["score_key"]
         and int(row["input"]["event_range"]["start"]) < current_start),
        key=lambda row: int(row["input"]["event_range"]["start"]),
    )
    active = False
    for row in earlier:
        actions = sorted(
            row.get("reference_plan", {}).get("actions", []),
            key=lambda action: int(action["source_index"]),
        )
        for action in actions:
            text = str(action.get("text") or action.get("jianzi_text") or "")
            for marker in re.findall(r"泛起|泛止", text):
                active = marker == "泛起"
    return active


def pitch_audit_notes(
    item: dict, actions_by_source: dict[int, dict],
    historical: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Build audit rows by replaying every confirmed earlier phrase."""
    handoff = (item.get("input") or {}).get("phrase_handoff") or {}
    previous = handoff.get("previous_phrase") or {}
    current_notes = list((item.get("input") or {}).get("notes_without_jianzi") or [])
    prefix_phrases: list[tuple[list[dict], list[dict]]] = []
    if historical:
        current_start = int(item["input"]["event_range"]["start"])
        earlier = sorted(
            (row for (score_key, _), row in historical.items()
             if score_key == item["score_key"]
             and int(row["input"]["event_range"]["start"]) < current_start),
            key=lambda row: int(row["input"]["event_range"]["start"]),
        )
        prefix_phrases = [
            (list(row["input"].get("notes_without_jianzi") or []),
             list(row.get("reference_plan", {}).get("actions") or []))
            for row in earlier
        ]
    if not prefix_phrases and previous:
        prefix_phrases = [(
            list(previous.get("notes") or []),
            list(previous.get("actions") or []),
        )]

    def render_rows(notes: list[dict], action_map: dict[int, dict]) -> list[dict]:
        rows = []
        for note in notes:
            row = deepcopy(note)
            action = action_map.get(int(note["index"]))
            if action is None:
                row["jianzi"] = ""
            else:
                value = action.get("jianzi_text")
                if value is None:
                    value = action.get("text") or ""
                row["jianzi"] = str(value)
            rows.append(row)
        return rows

    prefix = []
    # A phrase may begin in the middle of a harmonic passage.  The previous
    # phrase's structured action state then tells us the harmonic hui even if
    # its readable text omits ``泛起`` and the hui as an inherited field.
    all_prefix_actions = [action for _, actions in prefix_phrases for action in actions]
    seed_allowed = item.get("harmonic_region_at_start")
    if seed_allowed is None:
        seed_allowed = True
        for action in all_prefix_actions:
            text = str(action.get("text") or action.get("jianzi_text") or "")
            for marker in re.findall(r"泛起|泛止", text):
                seed_allowed = marker == "泛起"
    seed_hui = next(
        (action.get("hui") for action in reversed(all_prefix_actions)
         if seed_allowed and action.get("mode") == "harmonic"
         and action.get("hui") is not None),
        None,
    )
    if seed_hui is not None:
        prefix.append({
            "index": -1,
            "jianpu": None,
            "abc": "",
            "duration": "",
            "jianzi": f"泛起勾一弦{hui_label(seed_hui)}",
        })
    for notes, actions in prefix_phrases:
        action_map = {
            int(action["source_index"]): action
            for action in actions if action.get("source_index") is not None
        }
        prefix.extend(render_rows(notes, action_map))
    return prefix + render_rows(current_notes, actions_by_source)


def render_edit_preview(actions: list[dict], patches: list[dict], valid: bool,
                        errors: list[dict], item: dict | None = None,
                        warnings: list[dict] | None = None) -> str:
    changed = {int(patch["source_index"]) for patch in patches
               if patch.get("source_index") is not None}
    by_index = {int(action["source_index"]): action for action in actions
                if int(action["source_index"]) in changed}
    patch_types: dict[int, list[str]] = {}
    for patch in patches:
        patch_types.setdefault(int(patch["source_index"]), []).append(
            str(patch.get("patch_type") or "UNKNOWN"))
    mode_names = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
    lines = [f'预览｜{"可应用" if valid else "不可应用"}｜变更{len(changed)}音',
             "序号｜简谱｜减字显示｜谱面减字"]
    pitch_warnings = {
        int(warning["source_index"])
        for warning in (warnings or [])
        if warning.get("code") == "jianzi_pitch_mismatch"
        and warning.get("source_index") is not None
    }
    for index in sorted(changed):
        action = by_index.get(index)
        visible_index = event_index_for_source(item, index) if item is not None else None
        visible_index = index if visible_index is None else visible_index
        if action is None:
            lines.append(
                f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
                f'未找到动作｜-')
            continue
        mode = mode_names.get(action.get("mode"), action.get("mode") or "待定")
        partner = ""
        if isinstance(action.get("string2"), int):
            if action.get("mode2") == "open" and action.get("mode") != "open":
                partner = f'＋散{action["string2"]}弦'
            else:
                partner = f'＋{action["string2"]}弦'
        if action.get("mode") == "open":
            position = f'{action.get("string")}弦{partner}'
        else:
            position = f'{hui_label(action.get("hui"))}{action.get("string")}弦{partner}'
        left = action.get("left_finger") or "左手待定"
        right = action.get("right_finger") or ("续音" if not action.get("attack") else "右手待定")
        surface = render_jianzi_surface(
            action, omitted_placeholder=OMITTED_PLACEHOLDER
        ) or "—"
        explicit = action.get("jianzi_text")
        display = ("待填写" if explicit is None else
                   ("已置空" if explicit == "" else "已填写"))
        jianpu = pitch_label(item, index) if item is not None else ""
        warning_suffix = ":warning:音高不匹配" if index in pitch_warnings else ""
        lines.append(
            f'{visible_index}｜{jianpu}｜{display}｜{surface}{warning_suffix}'
        )
    if errors:
        lines.append("错误｜" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


class RealToolRuntime:
    def __init__(self, item: dict, historical: dict[tuple[str, str], dict], *,
                 basic_fingering: bool = False,
                 toward_reference: bool = False,
                 target_patches: list[dict] | None = None) -> None:
        self.item = item
        self.historical = historical
        self.basic_fingering = basic_fingering
        self.toward_reference = toward_reference
        self.target_patches = list(target_patches or [])
        self.calls: list[dict] = []
        self.event_to_source = {
            int(note["event_index"]): int(note["index"])
            for note in item.get("input", {}).get("notes_without_jianzi", [])
            if note.get("event_index") is not None
        }
        self.source_to_event = {source: event for event, source in self.event_to_source.items()}
        # edit_plan 的累计 patch：多批次提交时，预览与审计始终针对累计状态，
        # 否则后续批次会被误报为“其余音全部待定”。重复序号按序覆盖。
        self.accumulated_patches: list[dict] = []

    def _event_to_source(self, value: Any) -> int:
        """Resolve public continuous 音序; legacy source indexes remain accepted."""
        index = int(value)
        return self.event_to_source.get(index, index)

    def _source_to_event(self, value: Any) -> int:
        index = int(value)
        return self.source_to_event.get(index, index)

    def invoke(self, name: str, args: dict) -> dict:
        try:
            audit_result = None
            if name == "expand_context":
                key = (self.item["score_key"], str(args["phrase_id"]))
                previous = ((self.item.get("input") or {}).get("phrase_handoff") or {}).get("previous_phrase") or {}
                if str(previous.get("phrase_id") or "") == key[1]:
                    raise ValueError(
                        "expand_context must not repeat the read-only previous phrase already shown in user content"
                    )
                older = self.historical[key]
                if (int(older["input"]["event_range"]["start"]) >=
                        int(self.item["input"]["event_range"]["start"])):
                    raise ValueError("expand_context only permits earlier phrases")
                result = {
                    "phrase_id": key[1], "readonly": True,
                    "text": (
                        f"【只读更早段 {key[1]}｜confirmed_readonly】\n"
                        "序号｜简谱｜ABC｜时值｜谱面减字\n"
                        + _render_phrase_lines(
                            older["input"]["notes_without_jianzi"],
                            older["reference_plan"]["actions"], readonly=True,
                        )
                    ),
                }
            elif name == "list_context":
                current_start = int(self.item["input"]["event_range"]["start"])
                entries = []
                for (score_key, phrase_id), older in self.historical.items():
                    if score_key != self.item["score_key"]:
                        continue
                    if int(older["input"]["event_range"]["start"]) >= current_start:
                        continue
                    section = (older["input"].get("phrase_handoff") or {}).get("section") or {}
                    entries.append({"phrase_id": phrase_id,
                                    "section_marker": section.get("marker", ""),
                                    "event_range": older["input"]["event_range"]})
                entries.sort(key=lambda entry: int(entry["event_range"]["start"]))
                result = {"count": len(entries), "phrases": entries, "readonly": True}
            elif name == "get_pitch_candidates":
                explicit_target = args.get("target_midi")
                skipped_indices = []
                if explicit_target is not None:
                    groups = [{"target_midi": float(explicit_target), "sources": []}]
                    is_batch_query = False
                else:
                    if args.get("event_index") is not None:
                        indices = [self._event_to_source(args["event_index"])]
                    elif args.get("event_indices") is not None:
                        indices = [self._event_to_source(value) for value in args.get("event_indices") or []]
                    elif args.get("source_index") is not None:
                        indices = [int(args["source_index"])]
                    else:
                        indices = [int(value) for value in args.get("source_indices") or []]
                    if not indices:
                        raise ValueError("source_index, source_indices or target_midi is required")
                    is_batch_query = len(indices) > 1
                    valid_entries = []
                    skipped_indices = []
                    for index in indices:
                        try:
                            note = next(n for n in self.item["input"]["notes_without_jianzi"]
                                        if int(n["index"]) == index)
                            tones = abc_chord_tones(str(note.get("abc") or ""))
                            targets = target_midis(self.item, index)
                        except (StopIteration, ValueError):
                            skipped_indices.append(index)
                            continue
                        valid_entries.append((index, note, tones, targets))
                    if not valid_entries:
                        raise ValueError("no source index with a parseable sounding target")
                    grouped: dict[float, dict] = {}
                    for index, note, tones, targets in valid_entries:
                        # 和弦逐音查询：标签用各自 ABC 音名（简谱只承载单音）。
                        for position, target in enumerate(targets, 1):
                            label = None
                            if len(targets) > 1:
                                token = tones[position - 1][0] if position <= len(tones) else ""
                                label = f'和弦音{position}（{token}）' if token else f'和弦音{position}'
                            target_value = float(target)
                            key = round(target_value, 6)
                            group = grouped.setdefault(
                                key, {"target_midi": target_value, "sources": []}
                            )
                            group["sources"].append({
                                "event_index": self._source_to_event(index),
                                "source_index": index,
                                "jianpu": chord_jianpu_label(
                                    note.get("abc"),
                                    note.get("jianpu") or note.get("jianpu_alt") or "",
                                    note.get("jianpu_alt"),
                                ),
                                "label": label or f'简谱{note.get("jianpu") or note.get("jianpu_alt") or "未知"}',
                            })
                    groups = list(grouped.values())
                type_value = args.get("类型")
                requested_types = (type_value if isinstance(type_value, list)
                                   else [type_value])
                type_mapping = {"泛音": "harmonic", "散音": "open", "按音": "stopped"}
                type_modes = list(dict.fromkeys(
                    type_mapping[value] for value in requested_types
                    if value in type_mapping
                ))
                # 接受文档中的单字符串，也接受模型常见的字符串数组形式。
                modes = (tuple(type_modes) if type_modes else
                         tuple(args.get("modes") or ("stopped", "open", "harmonic")))
                audit_queries, tables = [], []
                for group in groups:
                    target = float(group["target_midi"])
                    sources = list(group.get("sources") or [])
                    index = (int(sources[0]["source_index"]) if sources
                             else args.get("source_index"))
                    event = ScoreEvent(
                        id=(f"src-{index:05d}" if index is not None else "pitch"),
                        kind="note", abc="",
                        pitches_midi=[round(target)], onset_ticks=0,
                        duration_ticks=960, bar=1, beat=1.0, attack=True,
                        section_id="sec", phrase_id=self.item["phrase_id"],
                    )
                    candidates = candidates_for_event(
                        event, self.item["input"]["normalized_tuning"]["open_midi"],
                        tolerance_cents=50.0, modes=modes,
                        # 批量查询默认 8 条候选，单音查询默认 12；显式传入优先。
                        max_candidates=int(args.get(
                            "max_candidates", 8 if is_batch_query else 12)),
                    )
                    candidate_rows = [candidate.to_dict() for candidate in candidates]
                    audit_queries.append({
                        "event_indices": list(dict.fromkeys(
                            int(source["event_index"]) for source in sources
                        )),
                        # Internal audit compatibility; this field is not
                        # included in the rendered tool observation.
                        "source_indices": list(dict.fromkeys(
                            int(source["source_index"]) for source in sources
                        )),
                        "sources": sources,
                        "target_midi": target,
                        "candidates": candidate_rows,
                    })
                    if sources:
                        tables.append(render_grouped_candidate_table(
                            target, candidate_rows, modes, sources))
                    else:
                        tables.append(render_pitch_candidate_table(
                            target, candidate_rows, modes))
                audit_result = {"queries": audit_queries}
                if skipped_indices:
                    tables.append(
                        "跳过不可解析或非发音序号｜"
                        + "、".join(str(self._source_to_event(index)) for index in dict.fromkeys(skipped_indices))
                    )
                result = {"text": "\n\n".join(tables)}
            elif name == "edit_plan":
                if "patches" in args:
                    raise ValueError("edit_plan only accepts jianzi_rows")
                patches = []
                current_replay = replay_patches(
                    self.item["baseline_plan"], self.accumulated_patches,
                    strict_before=False,
                )
                current_text = {
                    int(action["source_index"]): action.get("jianzi_text")
                    for action in current_replay.actions
                }
                current_actions = {
                    int(action["source_index"]): action
                    for action in current_replay.actions
                }
                skipped_not_editable: list[int] = []
                # Public rows are [event_index, text]. Convert to sparse
                # source indexes before replay; bars have no event index and
                # therefore fall into the existing non-editable warning path.
                public_rows = []
                unchanged_sources: list[int] = []
                for row in list(args.get("jianzi_rows") or []):
                    if isinstance(row, list) and row:
                        row = list(row)
                        try:
                            row[0] = self._event_to_source(row[0])
                        except (TypeError, ValueError):
                            pass
                        if (len(row) == 2 and isinstance(row[0], int)
                                and isinstance(row[1], str)
                                and row[0] in current_text
                                and current_text[row[0]] == row[1]):
                            unchanged_sources.append(row[0])
                    public_rows.append(row)
                patches.extend(expand_jianzi_rows(
                    public_rows, current_text=current_text,
                    allowed_indices=set(current_actions),
                    skipped_not_editable=skipped_not_editable,
                ))
                validate_patch_vocabulary(patches)
                batch_replay = replay_patches(self.item["baseline_plan"], patches,
                                              strict_before=False)
                batch_target_errors = (compare_replay_to_patch_targets(
                    batch_replay.actions, patches
                ) if batch_replay.valid else [])
                if batch_replay.valid and not batch_target_errors:
                    self.accumulated_patches.extend(patches)
                    cumulative = self.accumulated_patches
                else:
                    cumulative = None
                replay = (replay_patches(self.item["baseline_plan"], cumulative,
                                          strict_before=False)
                          if cumulative is not None else batch_replay)
                preview_valid = replay.valid
                preview_errors = list(replay.errors) + list(batch_target_errors)
                preview_warnings = ([{
                    "code": "ignored_noneditable_jianzi_rows",
                    "event_indices": [self._source_to_event(index)
                                      for index in skipped_not_editable],
                }] if skipped_not_editable else [])
                if replay.valid:
                    target_errors = compare_replay_to_patch_targets(
                        replay.actions, cumulative if cumulative is not None else patches
                    )
                    if target_errors:
                        preview_valid = False
                        preview_errors.extend(target_errors)
                if preview_valid and self.basic_fingering:
                    preview_valid, basic_report = validate_jianzi_only(
                        self.item, cumulative, toward_reference=False,
                        require_complete=False, historical=self.historical)
                    preview_errors = list(basic_report["problems"])
                    preview_warnings.extend(basic_report.get("warnings") or [])
                elif preview_valid and self.toward_reference:
                    # Public tool feedback must never compare against the
                    # teacher-only reference.  The teacher already has the
                    # private GQS for planning; exposing expected text here
                    # would copy the answer into the student trajectory.
                    preview_valid, quality_report = validate_jianzi_only(
                        self.item, cumulative, toward_reference=False,
                        require_complete=False, historical=self.historical)
                    preview_errors = list(quality_report["problems"])
                    preview_warnings.extend(quality_report.get("warnings") or [])
                audit_result = {
                    "valid": preview_valid,
                    "errors": preview_errors,
                    "warnings": preview_warnings,
                    "preview_actions": replay.actions,
                }
                # 公开结果只保留教师决策需要的信息；紧凑行无 patch_id，
                # applied_patch_ids 只是空串回执，公开与审计均不保留。
                preview_text = render_edit_preview(
                    replay.actions, cumulative or patches, preview_valid,
                    preview_errors, self.item, preview_warnings)
                unchanged_events = sorted({
                    self._source_to_event(index) for index in unchanged_sources
                })
                if unchanged_events:
                    preview_text += "\n未变更｜" + "、".join(
                        str(index) for index in unchanged_events
                    ) + "｜提交值与当前减字相同"
                remaining_warnings = [
                    warning for warning in preview_warnings
                    if warning.get("code") != "jianzi_pitch_mismatch"
                ]
                if remaining_warnings:
                    preview_text += "\n警告｜" + json.dumps(
                        remaining_warnings, ensure_ascii=False, separators=(",", ":"))
                result = {
                    "valid": preview_valid,
                    "text": preview_text,
                }
                if unchanged_events:
                    result["unchanged_event_indices"] = unchanged_events
            else:
                raise ValueError(f"unknown tool: {name}")
            envelope = {"ok": True, "result": result}
            audit_envelope = {"ok": True, "result": audit_result or result}
        except Exception as exc:
            envelope = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            audit_envelope = envelope
        self.calls.append({"name": name, "arguments": deepcopy(args),
                           "result": audit_envelope,
                           "public_result": envelope})
        return envelope


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


def parse_final(text: str) -> dict:
    """Extract the first complete JSON object from a model response.

    Models sometimes wrap the object in Markdown fences or a short prose
    prefix/suffix.  ``raw_decode`` lets us stop at the end of the first
    complete object instead of joining unrelated braces with ``rfind``.
    When the object is truncated only by missing closing delimiters, a
    conservative delimiter repair is attempted; no tokens are changed.
    """
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = text[start:]
        try:
            payload, _ = decoder.raw_decode(candidate)
            if isinstance(payload, dict):
                return payload
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
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            last_error = exc
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
            payload = parse_final(text)
        except ValueError as exc:
            payload = recover_explicit_noop_prose(text)
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
    if set(payload) - {"decision_summary", "tool_calls"} or "tool_calls" not in payload:
        raise ValueError("teacher envelope must contain only tool_calls and optional decision_summary")
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
    normalized = []
    for position, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            raise ValueError(f"tool_calls[{position}] must be an object")
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments")
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


def normalize_patches(patches: list[dict], *, prefix: str = "teacher") -> list[dict]:
    normalized = []
    for number, patch in enumerate(patches, 1):
        item = deepcopy(patch)
        item.setdefault("patch_id", f"{prefix}-{number:04d}")
        item.setdefault("before", {})
        item.setdefault("after", {})
        normalized.append(item)
    return normalized


def canonical_jianzi_text(text: str | None) -> str | None:
    """Normalize harmless surface variants before private reference review."""
    if text is None:
        return None
    digits = str.maketrans("一二三四五六七", "1234567")
    return re.sub(r"\s+", "", str(text)).replace("弦", "弦").translate(digits)


def infer_jianzi_text_patches(
    baseline: dict, references: list[dict | ReferenceAction]
) -> dict[str, list[dict]]:
    """Find Guqinizer targets from the written jianzi text only.

    Whether a second stage is needed is a surface-notation decision.  Do not
    infer it from mode/string/hui/finger fields: those fields may be partial,
    inherited, or marked conflicting even when the annotation contains an
    explicit text that should be reviewed.  A blank annotation is an explicit
    empty display target, regardless of parser confidence: surface comparison
    must be able to clear redundant glyphs covered by 吟/猱 or a compound
    multi-sound gesture.  Data-quality filtering handles wholly blank phrases
    before either model stage is generated.
    """
    baseline_by_index = {
        int(action["source_index"]): action
        for action in baseline.get("actions", [])
    }
    patches: list[dict] = []
    for reference in references:
        if isinstance(reference, dict):
            source_index = reference.get("source_index")
            confidence_class = reference.get("confidence_class", "weak")
            confidence = reference.get("confidence", 0.5)
            evidence = list(reference.get("evidence") or [])
            if reference.get("jianzi_text") is not None:
                expected = reference.get("jianzi_text")
            else:
                expected = reference.get("text")
        else:
            source_index = reference.source_index
            confidence_class = reference.confidence_class
            confidence = reference.confidence
            evidence = list(reference.evidence)
            expected = (reference.jianzi_text
                        if reference.jianzi_text is not None
                        else reference.text)
        try:
            source_index = int(source_index)
        except (TypeError, ValueError):
            continue
        # Reference parsing uses None for an ordinary blank source cell and
        # "" for a known compound-covered continuation.  At the surface layer
        # both mean the same target: render no glyph on this row.
        if expected is None:
            expected = ""
        if not isinstance(expected, str):
            continue
        before_action = baseline_by_index.get(source_index)
        if before_action is None:
            continue
        before = before_action.get("jianzi_text")
        if canonical_jianzi_text(before) == canonical_jianzi_text(expected):
            continue
        patches.append({
            "patch_id": f"text-target-{len(patches) + 1:04d}",
            "patch_type": "SET_JIANZI_TEXT",
            "source_index": source_index,
            "before": {"jianzi_text": before},
            "after": {"jianzi_text": expected},
            "confidence_class": confidence_class,
            "confidence": confidence,
            "evidence": evidence + ["jianzi_text_surface_diff"],
            "required_tools": [],
        })
    return {"patches": patches}


def is_all_empty_reference_phrase(item: dict) -> bool:
    """True when a phrase has no annotated glyph target at all.

    Such phrases provide no supervision for either Base/Fingering or
    Guqinizer and are excluded before API generation.  Materialized repeat
    rows are different: their visible target is the explicit repeat-omission
    placeholder, so a phrase containing one is not classified as all-empty.
    """
    notes = item.get("input", {}).get("notes_without_jianzi", [])
    if any(note.get("notation_omitted") for note in notes):
        return False
    # v9 repeat-semantics inference keeps repeat markers in provenance rather
    # than copying them into ``notation_omitted`` on each note.  Such a phrase
    # can legitimately have an all-empty visible target while still carrying
    # educational information (e.g. a ``再作``/``从ㄱ再作`` omission).  Keep it
    # eligible for the reasoned no-op final reply pass.
    repeat = item.get("provenance", {}).get("repeat_materialization", {}) or {}
    if repeat.get("markers") or repeat.get("marker_only_notes") or repeat.get("copy_markers"):
        return False
    references = item.get("reference_plan", {}).get("actions", [])
    for reference in references:
        value = (reference.get("jianzi_text")
                 if reference.get("jianzi_text") is not None
                 else reference.get("text"))
        if isinstance(value, str) and value.strip():
            return False
    return True


def expand_jianzi_rows(
    rows: list, *, current_text: dict[int, str | None] | None = None,
    allowed_indices: set[int] | None = None,
    skipped_not_editable: list[int] | None = None,
) -> list[dict]:
    """Expand concise ``[source_index, jianzi_text]`` rows into typed patches."""
    expanded = []
    seen = set()
    not_editable: list[int] = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"jianzi_rows[{number}] must be [source_index, jianzi_text]")
        index, text = row
        # Models occasionally serialize a JSON integer as a quoted decimal.
        # This is a recoverable transport error, unlike floats/booleans or
        # arbitrary strings, so normalize it before validation.
        if isinstance(index, str) and re.fullmatch(r"[0-9]+", index):
            index = int(index)
        if (not isinstance(index, int) or isinstance(index, bool)
                or not isinstance(text, str)):
            raise ValueError(f"jianzi_rows[{number}] has invalid types")
        if index in seen:
            raise ValueError(f"duplicate jianzi source_index: {index}")
        if allowed_indices is not None and index not in allowed_indices:
            # Do not abort on the first bad row.  Models often submit a
            # broad range containing bars/rests; returning the complete set
            # lets the next turn correct the whole batch in one pass.
            not_editable.append(index)
            seen.add(index)
            continue
        seen.add(index)
        before = (current_text or {}).get(index)
        if current_text is not None and before == text:
            continue
        expanded.append({"patch_type": "SET_JIANZI_TEXT", "source_index": index,
                         "before": {"jianzi_text": before},
                         "after": {"jianzi_text": text}})
    if not_editable and skipped_not_editable is not None:
        skipped_not_editable.extend(not_editable)
    elif not_editable:
        indices = ", ".join(str(index) for index in not_editable)
        raise ValueError(f"jianzi source_index is not editable: {indices}")
    return expanded


def validate_patch_vocabulary(patches: list[dict]) -> None:
    """The public edit protocol may only produce final jianzi text patches."""
    for patch in patches:
        if patch.get("patch_type") != "SET_JIANZI_TEXT":
            raise ValueError("edit_plan only accepts SET_JIANZI_TEXT patches")


def validate_jianzi_only(
    item: dict, patches: list[dict], *, toward_reference: bool = False,
    require_complete: bool = True,
    historical: dict[tuple[str, str], dict] | None = None,
) -> tuple[bool, dict]:
    """Validate the minimal text-only protocol.

    Syntax/coverage and safe reference omissions are hard conditions. Pitch is
    computed from the authored notation itself and is always advisory during
    the model conversation; the same report exposes a private eligibility bit
    for filtering training exports.
    """
    patches = normalize_patches(patches, prefix="jianzi")
    illegal = [patch for patch in patches
               if patch.get("patch_type") != "SET_JIANZI_TEXT"]
    problems = ([{"code": "non_jianzi_patch_in_minimal_protocol",
                  "patch_type": patch.get("patch_type")} for patch in illegal])
    replay = replay_patches(item["baseline_plan"], patches, strict_before=False)
    problems.extend(replay.errors)
    by_index = {int(action["source_index"]): action for action in replay.actions}
    if require_complete:
        for action in replay.actions:
            if action.get("jianzi_text") is None:
                problems.append({"source_index": int(action["source_index"]),
                                 "code": "pending_jianzi_text"})

    reference_review: list[dict] = []
    if toward_reference:
        # A Base/Fingering repeat-omission marker is an explicit display
        # value, not an ordinary empty target.  Guqinizer may leave it
        # unchanged, but must not erase it by submitting an empty string.
        baseline_by_index = {
            int(action["source_index"]): action.get("jianzi_text")
            for action in (item.get("baseline_plan", {}).get("actions") or [])
        }
        reference_by_index = {
            int(action["source_index"]): (
                action.get("jianzi_text")
                if action.get("jianzi_text") is not None
                else action.get("text")
            )
            for action in (item.get("reference_plan", {}).get("actions") or [])
        }
        for source_index, baseline_text in baseline_by_index.items():
            if (baseline_text == OMITTED_PLACEHOLDER
                    and reference_by_index.get(source_index) == OMITTED_PLACEHOLDER
                    and by_index.get(source_index, {}).get("jianzi_text") == ""):
                problems.append({
                    "source_index": source_index,
                    "code": "repeat_omission_marker_cleared",
                })
        for reference in item.get("reference_plan", {}).get("actions", []):
            if reference.get("confidence_class") not in {"verified", "weak"}:
                continue
            action = by_index.get(int(reference["source_index"]))
            if action is None:
                continue
            expected = reference.get("jianzi_text")
            actual = action.get("jianzi_text")
            equivalent = canonical_jianzi_text(actual) == canonical_jianzi_text(expected)
            if (reference.get("confidence_class") == "verified"
                    and expected is not None and not equivalent):
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "reference_jianzi_text_mismatch",
                    "expected": expected, "actual": actual,
                })
            elif bool(expected) and actual == "":
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "nonempty_reference_jianzi_dropped",
                    "expected": expected,
                })

    warnings: list[dict] = []
    pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                    "details": []}
    try:
        current_source_indices = {
            int(note["index"])
            for note in item["input"].get("notes_without_jianzi", [])
            if note.get("index") is not None
        }
        notes = pitch_audit_notes(item, by_index, historical=historical)
        pitch_report = AUDIT.audit({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }, 50.0)
        for detail in pitch_report.get("details", []):
            if (int(detail.get("index")) in current_source_indices
                    and detail.get("status") == "mismatched"):
                warnings.append({
                    "source_index": detail.get("index"),
                    "code": "jianzi_pitch_mismatch",
                    "pairs": detail.get("pairs") or [],
                })
    except Exception as exc:
        # Tool/parser availability never blocks creative notation generation.
        pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                        "details": [], "unavailable": f"{type(exc).__name__}: {exc}"}

    current_pitch_details = [
        detail for detail in pitch_report.get("details", [])
        if detail.get("index") is not None
        and int(detail["index"]) in current_source_indices
    ]
    current_mismatch_count = sum(
        detail.get("status") == "mismatched" for detail in current_pitch_details
    )
    current_counts = {
        status: sum(detail.get("status") == status for detail in current_pitch_details)
        for status in ("matched", "mismatched", "skipped")
    }
    pitch_report["current_phrase_summary"] = {
        "total_notes": len(current_pitch_details),
        **current_counts,
        "match_rate": (
            round(current_counts["matched"] /
                  max(current_counts["matched"] + current_counts["mismatched"], 1), 6)
            if current_counts["matched"] + current_counts["mismatched"] else None
        ),
    }
    return not problems, {
        "valid": not problems,
        "problems": problems,
        "warnings": warnings,
        "private_reference_review": reference_review,
        "plan": {"actions": replay.actions},
        "patches": patches,
        "pitch_audit": pitch_report,
        "offline_quality": {
            # Pitch comparison is advisory.  A reliably parsed mismatch must
            # be shown to the teacher, but must not reject an otherwise valid
            # edit or silently remove the trajectory from training output.
            "training_eligible": not problems,
            "high_confidence_pitch_mismatches": current_mismatch_count,
        },
    }


def can_accept_empty_tool_turn(item: dict, targets: list[dict]) -> bool:
    """Return whether an empty tool call can legitimately finish a stage.

    A Guqinizer stage with inferred surface-text targets must not be accepted
    as a no-op merely because the Base plan is complete.  The target list is
    computed before the model call and is the authoritative signal that a
    second-stage edit is required.  True no-op reviews pass an empty target
    list and are still checked against the complete baseline below.
    """
    if targets:
        return False
    return validate_jianzi_only(item, [], toward_reference=False)[0]


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
    basic = objective == "basic_fingering"
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
            "为当前段每个演奏事件填写可读、可演奏的基础减字初稿"
            if basic else
            ("复核现有减字，并说明当前稿为何可以直接保留"
             if review_noop else
             "依据 GQS 与上下文润色最终减字、复杂技法和空显示范围")
        )
    }
    private_instruction["rules"] = ([
        "逐音或按相邻音组分析现有取音、衔接、技法和显示范围为什么合理；只评价当前方案，不提出任何改写、补充、置空或省略建议。",
        f"直接给出可独立核查的音乐学复核意见，不调用工具；最后必须用“{NOOP_CONCLUSION}”收束。",
        "用“当前稿”或“当前谱面”指称现有内容，不讨论生成流程或数据来源。",
    ] if review_noop else [
        "唯一编辑入口是 edit_plan.jianzi_rows=[[音序,减字文字],...]；小节线没有音序。",
        "私有标注可以用于内部决定哪里需要改、往什么方向改；decision_summary 会原样进入学生可见轨迹，公开 reasoning 的任务是解释修改为什么在古琴演奏上合理，而不是隐藏标注事实后重新证明答案。",
        *([] if allow_private_reasoning_leakage else [
            "decision_summary 绝对不得提及 GQS、教师提示、系统提示、教师私有参考、最终标注、参考答案或目标答案；不得写“与参考一致”“参考使用”或“按提示”；也不得转述只有私有标注中出现的具体减字作为理由。需要使用私有目标规划时，只在内部决定 tool_calls，不在摘要中说明来源。"
        ]),
        "最终结束前每个演奏事件必须得到字符串；一次 edit_plan 可以只提交已决定修改的部分行，未提交的行保留现状。空字符串的直接语义是将该行 jianzi_text 置空；不会删除声音或演奏状态。若前一减字已覆盖多个动作，后续行可用空字符串表示不重复显示。",
        "音高工具只在减字与简谱都可可靠解析时给出警告；复杂技法或未解析动作不算失败。",
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
    if basic:
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
            "必须从头到尾按音序审阅完整个当前段：可以逐音分析，也可以每次按一小组相邻音分析，但不能只讨论少数差异音便结束。私有标注用于提示值得关注的位置和方向，不要求逐字照抄；公开 reasoning 应说明各音或各小组为何保留或修改，以及相关的音高、技法和减字取舍，不要说“因为标注/参考这样写”。edit_plan.jianzi_rows 只填写实际决定改写的音，不必重发保留项。",
            "遇到复杂技法时，应结合相邻音组说明其演奏前提和效果，例如承接的弦、按位、右手动作或余音状态；重点核对走手动作的同弦连续性，以及掐撮三声等组合技法此前是否已建立所需的撮弦、按散关系和按位。允许依据完整上下文作出与私有标注不同但合理的判断。",
            "如果逐音分析后确认当前基础初稿已经合适，不需要润色，则不要调用 edit_plan；tool_calls 返回空数组，并在 decision_summary 中逐音或按相邻音组说明为何现有指法、衔接和技法已经足够，无需为了制造变化而改写。",
        ] + private_reference_semantics_rules(stage) + private_instruction["rules"]
    # Retry diagnostics remain in the private failure trace only.  Do not add
    # them to the teacher prompt: they can distract the model and expose
    # implementation details unrelated to the musical decision.
    if toward:
        private_instruction["rules"].extend(render_compound_gesture_knowledge(item))
    private_instruction["output_contract"] = ({
        "final_answer": f"直接输出最终复核意见文字，并以“{NOOP_CONCLUSION}”结尾；不要输出 JSON、tool_calls、Markdown 或代码围栏。",
    } if review_noop else {
        "tool_turn": {
            "decision_summary": "必填；基于公开谱面、前文或已有工具结果的理由或思考过程"
                               "（不得含未转义的半角双引号或换行，引用参数用中文引号）",
            "tool_calls": [{"name": "工具名", "arguments": {"参数": "值"}}],
        },
        "rules": [
            "每轮只输出一个 JSON 对象，不要输出 Markdown、代码围栏或 JSON 之外的文字。",
            "每次工具调用前必须填写 decision_summary；把一个或多个调用写入 tool_calls。",
            "若分析后确定无需调用工具，也必须填写有音乐学内容的 decision_summary，并令 tool_calls=[]。",
            "工具 Schema 已在 tools 字段中给出；不得编造工具名或参数。",
        ],
        "forbidden": (
            "decision_summary 不得提及 GQS、reference、teacher、target、教师提示、系统提示、提示词、教师私有参考、最终标注、标注答案或私有答案，"
            "不得把私有 GQS 中的具体减字内容包装成公开依据，"
            "不得复述隐藏推理过程。"
        ),
    })
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
                edited_events = {
                    int(row[0]) for row in arguments.get("jianzi_rows") or []
                    if isinstance(row, list) and row
                    and isinstance(row[0], int) and not isinstance(row[0], bool)
                }
                candidate_informed_edits = {
                    event_index for event_index in edited_events
                    if pitch_candidate_round.get(event_index, round_number) < round_number
                }
                pitch_repair_attempted.update(candidate_informed_edits)
                current_warning_events = {
                    runtime._source_to_event(int(warning["source_index"]))
                    for warning in audit_payload.get("warnings") or []
                    if warning.get("code") == "jianzi_pitch_mismatch"
                    and warning.get("source_index") is not None
                }
                # Once a warning has appeared, merely overwriting the row or
                # making it disappear in another same-turn edit is not enough:
                # the training trace must show a later candidate lookup and a
                # candidate-informed repair edit for that exact event.
                previously_pending = set(pending_pitch_warning_events)
                pending_pitch_warning_events.update(current_warning_events)
                pending_pitch_warning_events.difference_update(
                    (candidate_informed_edits & previously_pending)
                    - current_warning_events
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
                if pending_pitch_warning_events:
                    accepted_preview = None
                elif preview_matches:
                    accepted_preview = normalize_patches(accumulated)
        public_assistant = {"role": "assistant", "content": visible_summary}
        if public_calls:
            public_assistant["tool_calls"] = public_calls
        public_messages.append(public_assistant)
        public_messages.extend(public_results)
        if not teacher_calls:
            # An empty call list is a legitimate no-op only when the current
            # baseline is already complete (or the phrase has no editable
            # sounding events).  A blank fingering scaffold remains invalid,
            # so the model gets another turn instead of silently accepting it.
            if can_accept_empty_tool_turn(item, targets):
                accepted_preview = []
                no_edit_accepted = True
                final_payload = {"patches": []}
                break
            api_messages.append({
                "role": "user",
                "content": json.dumps({
                    "tool_results": [{
                        "ok": False,
                        "error": "当前段仍有待填写的演奏音，不能以空工具调用结束；请提交需要修改的 jianzi_rows。",
                    }],
                    "instruction": "根据这些真实工具结果继续；需要工具时输出 tool_calls。",
                }, ensure_ascii=False),
            })
            continue
        if accepted_preview is not None:
            final_payload = {"patches": accepted_preview}
            public_messages.append({
                "role": "assistant",
                "content": (
                    "工具预览已通过，当前段基础减字填写完成。"
                    if basic else "工具预览已通过，当前段减字润色完成。"
                ),
            })
            break
        if pending_pitch_warning_events:
            warning_indices = sorted(pending_pitch_warning_events)
            queried = all(
                index in pitch_candidate_round
                for index in warning_indices
            )
            if queried:
                instruction = (
                    "编辑工具仍报出音高不匹配。你已经查看过这些音的候选；"
                    f"请根据候选对音序 {warning_indices} 至少进行一次修复性 edit_plan，"
                    "只提交需要变化的行。即使判断警告可能来自复杂技法，也必须先尝试修复。"
                )
            else:
                instruction = (
                    "编辑工具报出音高不匹配，暂不能结束。先查看这些音的候选："
                    f"音序 {warning_indices}；下一轮调用 get_pitch_candidates，"
                    "把这些音序放入 event_indices。看到真实候选后，再用 edit_plan 尝试修复。"
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
    """Serialize API calls and enforce a minimum gap between requests."""

    def __init__(self, messages, min_interval: float):
        self._messages = messages
        self._min_interval = max(0.0, float(min_interval))
        self._last_call = 0.0

    def create(self, **options):
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
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
                      f"after {delay:.1f}s", flush=True)
                time.sleep(delay)
    raise RuntimeError("teacher attempts exhausted: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ABC_J/agent_training/inferred_v6/inferred_trajectories_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages_teacher_tools")
    parser.add_argument("--limit", type=int, default=2)
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
    # ``args.limit`` applies to the already-filtered eligible pool.  Do not
    # subtract global completed IDs here: Guqinizer resume commonly receives
    # a large fingering intermediate file containing no-op phrases outside
    # the requested retry set, and counting those would make the retry pool
    # appear empty.
    remaining = args.limit
    selected = pool[:remaining]
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
        for item_number, item in enumerate(selected, 1):
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
