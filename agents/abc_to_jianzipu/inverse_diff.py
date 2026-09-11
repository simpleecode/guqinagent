from __future__ import annotations

from typing import Any

from .inverse_models import PlanPatch, ReferenceAction


def infer_minimal_patches(baseline: dict, references: list[ReferenceAction]) -> dict:
    baseline_by_index = {int(item["source_index"]): item for item in baseline.get("actions", [])}
    patches: list[PlanPatch] = []
    unresolved: list[dict[str, Any]] = []
    patch_number = 0
    jianzi_text_state = {
        index: action.get("jianzi_text") for index, action in baseline_by_index.items()
    }

    def add(patch_type, reference, before, after, tools=None):
        nonlocal patch_number
        patch_number += 1
        patches.append(PlanPatch(
            patch_id=f"patch-{patch_number:04d}", patch_type=patch_type,
            source_index=reference.source_index, before=before, after=after,
            confidence_class=reference.confidence_class, confidence=reference.confidence,
            evidence=list(reference.evidence), required_tools=list(tools or []),
        ))

    for reference in references:
        if reference.confidence_class in {"conflicting", "unusable"}:
            unresolved.append({
                "source_index": reference.source_index,
                "confidence_class": reference.confidence_class,
                "reference": reference.to_dict(),
                "reason": "reference_not_safe_for_standard_sft",
            })
            continue
        before = baseline_by_index.get(reference.source_index)
        changed = False
        if reference.jianzi_text is not None:
            if before is None:
                unresolved.append({
                    "source_index": reference.source_index,
                    "confidence_class": reference.confidence_class,
                    "reference": reference.to_dict(),
                    "reason": "jianzi_text_has_no_playable_baseline_event",
                })
            else:
                current_text = jianzi_text_state.get(reference.source_index)
                if current_text != reference.jianzi_text:
                    add("SET_JIANZI_TEXT", reference,
                        {"jianzi_text": current_text},
                        {"jianzi_text": reference.jianzi_text})
                    jianzi_text_state[reference.source_index] = reference.jianzi_text
                    changed = True
            if reference.jianzi_text == "":
                continue
        if before is None:
            # Control/ornament events may intentionally have no baseline attack.
            if reference.techniques:
                for technique_number, technique in enumerate(reference.techniques):
                    after = {"technique": technique}
                    if technique_number == 0:
                        # The ADD_TECHNIQUE replay path can materialize a
                        # continuation-only row or a complete pre-attack row.
                        # Carry the available reference snapshot so the result
                        # renders and audits as the original action.
                        after["attack"] = reference.attack
                        for field in (
                            "mode", "string", "hui", "left_finger", "right_finger",
                            "string2", "mode2", "hui2",
                        ):
                            value = getattr(reference, field)
                            if value is not None:
                                after[field] = value
                    if technique in reference.pre_attack_techniques:
                        after["placement"] = "pre_attack"
                    add("ADD_TECHNIQUE", reference, {}, after)
            elif reference.text:
                unresolved.append({
                    "source_index": reference.source_index,
                    "confidence_class": reference.confidence_class,
                    "reference": reference.to_dict(),
                    "reason": "reference_action_has_no_baseline_event",
                })
            continue
        if reference.attack != bool(before.get("attack", True)):
            before_attack = {"attack": before.get("attack")}
            after_attack = {"attack": reference.attack}
            if not reference.attack:
                before_attack["right_finger"] = before.get("right_finger")
                after_attack["right_finger"] = None
            add("CHANGE_ATTACK", reference, before_attack, after_attack)
            changed = True
        if (reference.compound_gesture
                and reference.compound_gesture != before.get("compound_gesture")):
            add(
                "SET_COMPOUND_GESTURE", reference,
                {"compound_gesture": before.get("compound_gesture")},
                {"compound_gesture": reference.compound_gesture},
            )
            changed = True
        if reference.mode and reference.mode != before.get("mode"):
            add(
                "CHANGE_MODE", reference, {"mode": before.get("mode")},
                {"mode": reference.mode},
                [{"name": "calculate_guqin_pitch", "arguments": {
                    "source_index": reference.source_index, "mode": reference.mode,
                    "string": reference.string, "hui": reference.hui,
                }}],
            )
            changed = True
        if reference.string is not None and (
            reference.string != before.get("string")
            or (reference.hui is not None and reference.hui != before.get("hui"))
        ):
            add(
                "REPOSITION", reference,
                {"string": before.get("string"), "hui": before.get("hui")},
                {"string": reference.string, "hui": reference.hui},
                [
                    {"name": "get_pitch_candidates", "arguments": {
                        "source_index": reference.source_index,
                    }},
                    {"name": "calculate_guqin_pitch", "arguments": {
                        "source_index": reference.source_index, "mode": reference.mode,
                        "string": reference.string, "hui": reference.hui,
                    }},
                ],
            )
            changed = True
        elif reference.hui is not None and reference.hui != before.get("hui"):
            # A continuation such as “绰上五徽” may omit its inherited string
            # while stating a definite destination. Preserve that destination
            # without inventing a new string or requiring scalar pitch proof.
            add(
                "REPOSITION", reference,
                {"hui": before.get("hui")}, {"hui": reference.hui},
            )
            changed = True
        finger_after = {}
        finger_before = {}
        for field in ("left_finger", "right_finger"):
            value = getattr(reference, field)
            if value and value != before.get(field):
                finger_before[field] = before.get(field)
                finger_after[field] = value
        if finger_after:
            add("CHANGE_FINGER", reference, finger_before, finger_after)
            changed = True
        for technique in reference.techniques:
            after = {"technique": technique}
            if technique in reference.pre_attack_techniques:
                after["placement"] = "pre_attack"
            add("ADD_TECHNIQUE", reference, {}, after)
            changed = True
        if not changed and reference.attack:
            add("NO_OP", reference, before, before)
    return {
        "patches": [patch.to_dict() for patch in patches],
        "unresolved_differences": unresolved,
        "summary": {
            "patches": len(patches),
            "unresolved": len(unresolved),
            "verified_patches": sum(p.confidence_class == "verified" for p in patches),
            "weak_patches": sum(p.confidence_class == "weak" for p in patches),
        },
    }
