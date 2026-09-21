from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from .compound_gestures import find_compound_gesture, remove_compound_gesture
from .inverse_models import ReferenceAction


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
_SPEC = importlib.util.spec_from_file_location("inverse_reference_audit", AUDIT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(AUDIT)

RIGHT_HAND = tuple(AUDIT.RIGHT_HAND_TECHNIQUES)
LEFT_FINGER = tuple(AUDIT.LEFT_HAND_FINGERS)
TECHNIQUES = (
    "进复", "退复", "泛起", "泛止", "掐起", "带起", "抓起", "爪起", "滔起", "往来",
    "绰", "注", "吟", "猱", "撞", "逗", "撮", "泼", "剌", "进", "退", "复",
)
CONTROL_ONLY = {
    "泛止", "吟", "猱", "撞", "逗", "往来", "滔起", "进", "退", "复", "进复", "退复",
}
LEFT_HAND_ATTACKS = {"掐起", "带起", "抓起", "爪起", "罨", "虚罨", "推出"}


def _first(text: str, values: tuple[str, ...]) -> str | None:
    return next((item for item in values if item in text), None)


def _techniques(text: str) -> list[str]:
    remaining = text
    result: list[str] = []
    for token in sorted(TECHNIQUES, key=len, reverse=True):
        if token in remaining:
            result.append(token)
            remaining = remaining.replace(token, "")
    return result


def derive_jianzi_semantics(text: str) -> dict:
    """Derive audit-only structure from Agent-authored reduced notation.

    The notation text is the single editable source of truth for ornaments.
    An empty string deliberately means "render nothing" and must not erase
    the inherited/playable action underneath it.
    """
    raw = str(text or "").strip().strip("[]")
    if not raw:
        return {}
    compound = find_compound_gesture(raw)
    ordinary = remove_compound_gesture(raw) if compound else raw
    techniques = _techniques(ordinary)
    right = _first(ordinary, RIGHT_HAND)
    prefixes: list[str] = []
    if right:
        right_position = ordinary.find(right)
        prefixes = [token for token in techniques
                    if 0 <= ordinary.find(token) < right_position]
    return {
        "techniques": techniques,
        "pre_attack_techniques": prefixes,
    }


def parse_reference_actions(data: dict, audit_report: dict | None = None) -> list[ReferenceAction]:
    """Expand the pitch-relevant inheritance state without inventing intent."""
    details = {
        item.get("index"): item for item in (audit_report or {}).get("details", [])
    }
    context = AUDIT.new_context()
    result: list[ReferenceAction] = []
    harmonic = False
    for note in data.get("notes", []):
        index = int(note.get("index", len(result)))
        text = str(note.get("jianzi", "")).strip()
        jianzi_text = note.get("jianzi_text")
        compound_gesture = find_compound_gesture(text)
        ordinary_text = remove_compound_gesture(text) if compound_gesture else text
        detail = details.get(index, {})
        explicit: list[str] = []
        inherited: list[str] = []
        diagnostics: list[str] = []
        # Parse any explicit base pluck separately, but never leak the 撮/剌
        # substring inside 掐撮/掐拨剌 into ordinary double-stop rules.
        techniques = _techniques(ordinary_text)
        strings = [int(value) for value in re.findall(r"([1-7])弦", ordinary_text)]
        string = strings[0] if len(strings) == 1 else None
        string2 = None
        mode2 = None
        if string is not None:
            explicit.append("string")
        elif len(strings) > 1:
            # 双弦技法（撮／泼／剌）的主弦＋伙伴弦；两个位置各占一弦。
            dual = any(mark in ordinary_text for mark in ("撮", "泼", "剌"))
            if dual:
                string, string2 = strings[0], strings[1]
                explicit.append("string")
                explicit.append("string2")
                if re.search(r"[＋+][^＋+]*散", ordinary_text):
                    mode2 = "open"
                    explicit.append("mode2")
                elif re.search(r"[＋+][^＋+]*按音", ordinary_text):
                    mode2 = "stopped"
                    explicit.append("mode2")
            else:
                diagnostics.append("multiple_strings_require_compound_parser")
        hui = AUDIT.parse_hui(ordinary_text)
        if hui is not None:
            explicit.append("hui")
        left = _first(ordinary_text, LEFT_FINGER)
        if left:
            explicit.append("left_finger")
        right = _first(ordinary_text, RIGHT_HAND)
        if right:
            explicit.append("right_finger")
        elif string is not None and context.get("right_hand"):
            right = context["right_hand"]
            inherited.append("right_finger")
        pre_attack_techniques: list[str] = []
        if right:
            right_position = ordinary_text.find(right)
            pre_attack_techniques = [
                technique for technique in techniques
                if 0 <= ordinary_text.find(technique) < right_position
            ]
            if pre_attack_techniques:
                explicit.append("pre_attack_techniques")

        starts_harmonic = "泛起" in text
        ends_harmonic = "泛止" in text
        if starts_harmonic:
            harmonic = True
        # 双弦技法的“＋伙伴弦”段落只描述伙伴位；主位证据只看分隔符之前，
        # 否则伙伴的“散音”会把整个按音撮误判为散音。
        main_segment = re.split(r"[＋+]", ordinary_text, maxsplit=1)[0]
        explicit_open = "散" in main_segment and "泛起" not in text
        # 明文“按音”优先于泛音段上下文继承：标注写按音就是按音。
        text_says_stopped = "按音" in main_segment
        explicit_stopped = hui is not None or left is not None or text_says_stopped
        if (harmonic or "泛起" in text) and not text_says_stopped and not explicit_open:
            mode = "harmonic"
            if "泛" in main_segment:
                explicit.append("mode")
            else:
                inherited.append("mode")
        elif explicit_open:
            mode = "open"
            explicit.append("mode")
        elif explicit_stopped:
            mode = "stopped"
            explicit.append("mode")
        elif context.get("sound_mode") == "open" and string is not None:
            mode = "open"
            inherited.append("mode")
        elif string is not None or techniques:
            mode = "stopped"
            inherited.append("mode")
        else:
            mode = None

        if mode == "stopped":
            if hui is None and context.get("active_left_hui") is not None:
                # 徽外/徽外半 are symbolic positions, not floats.  Preserve
                # their label when the following reduced notation inherits
                # the stopped position.
                hui = context["active_left_hui"]
                inherited.append("hui")
            if left is None and context.get("left_finger"):
                left = context["left_finger"]
                inherited.append("left_finger")
            if string is None and techniques and context.get("active_left_string"):
                string = int(context["active_left_string"])
                inherited.append("string")
        if mode == "harmonic" and hui is None and context.get("harmonic_hui") is not None:
            hui = context["harmonic_hui"]
            inherited.append("hui")

        expected_sounding = any(
            AUDIT.parse_jianpu(note.get(field), AUDIT.parse_tonic_midi(data["metadata"]))
            is not None for field in ("jianpu", "jianpu_alt")
        )
        attack = bool(
            compound_gesture or right or strings
            or any(technique in LEFT_HAND_ATTACKS for technique in techniques)
        ) and text not in CONTROL_ONLY
        if techniques:
            # Whether a technique row contains a new right-hand attack is an
            # explicit structural fact, independent of scalar pitch confidence.
            explicit.append("attack")
        status = detail.get("status")
        reason = detail.get("reason")
        evidence: list[str] = []
        if jianzi_text == "":
            confidence_class, confidence = "verified", 1.0
            evidence.append("explicit_empty_jianzi_text")
        elif compound_gesture:
            # The gesture name is explicit, strong symbolic evidence even
            # though its multi-sound pitch contour is not scalar-auditable.
            confidence_class, confidence = "verified", 0.98
            evidence.append("explicit_context_dependent_compound_gesture")
            explicit.append("compound_gesture")
            explicit.append("attack")
        elif reason == "context_dependent_pre_attack_technique":
            confidence_class, confidence = "verified", 0.98
            evidence.append("explicit_pre_attack_technique")
        elif status == "matched":
            confidence_class, confidence = "verified", 0.98
            evidence.append("pitch_audit_matched")
        elif status == "mismatched":
            confidence_class, confidence = "conflicting", 0.1
            evidence.append("pitch_audit_mismatched")
        elif not text and expected_sounding:
            confidence_class, confidence = "unusable", 0.0
            diagnostics.append("empty_reference_for_sounding_event")
        elif reason in {"ornament_or_control", "context_dependent_slide", "context_dependent_transition"}:
            confidence_class, confidence = "weak", 0.65
            evidence.append(str(reason))
        elif expected_sounding:
            confidence_class, confidence = "weak", 0.45
            if reason:
                diagnostics.append(str(reason))
        else:
            confidence_class, confidence = "weak", 0.5

        if explicit:
            evidence.append("explicit_fields:" + ",".join(sorted(set(explicit))))
        if inherited:
            evidence.append("inherited_fields:" + ",".join(sorted(set(inherited))))
        result.append(ReferenceAction(
            source_index=index, text=text, attack=attack, mode=mode, string=string,
            hui=hui, left_finger=left, right_finger=right,
            string2=string2, mode2=mode2,
            compound_gesture=compound_gesture,
            jianzi_text=jianzi_text,
            techniques=techniques,
            pre_attack_techniques=pre_attack_techniques,
            explicit_fields=sorted(set(explicit)), inherited_fields=sorted(set(inherited)),
            confidence_class=confidence_class, confidence=confidence,
            evidence=evidence, diagnostics=diagnostics,
        ))

        # Reuse the mature audit state machine for the next action.
        if ordinary_text.strip():
            AUDIT.parse_jianzi(
                ordinary_text, AUDIT.parse_open_midi(data["metadata"]), context
            )
        # “泛止” is commonly attached to the final sounding glyph (for
        # example “3弦\n泛止”), not stored as a standalone row.  That final
        # glyph still belongs to the harmonic phrase; only following rows are
        # outside the scope.
        if ends_harmonic:
            harmonic = False
        if "休止" in str(note.get("jianpu", "")) and not text:
            context = AUDIT.new_context()
            harmonic = False
    return result
