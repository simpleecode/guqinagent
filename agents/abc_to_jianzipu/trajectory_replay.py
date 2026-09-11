from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


REPEAT_OMISSION_TEXT = "无（由于是再作部分，省略）"

from .reference_parser import derive_jianzi_semantics


STATE_FIELDS = (
    "mode", "string", "hui", "left_finger", "right_finger", "attack",
    "string2", "mode2", "hui2", "compound_gesture",
    "pre_attack_techniques", "jianzi_text",
)


@dataclass(slots=True)
class ReplayResult:
    valid: bool
    actions: list[dict[str, Any]]
    applied_patch_ids: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "actions": self.actions,
            "applied_patch_ids": self.applied_patch_ids,
            "errors": self.errors,
        }


def _error(patch: dict, code: str, **details: Any) -> dict[str, Any]:
    return {
        "patch_id": patch.get("patch_id"),
        "source_index": patch.get("source_index"),
        "code": code,
        **details,
    }


def replay_patches(
    baseline_plan: dict[str, Any],
    patches: Iterable[dict[str, Any]],
    *,
    patch_ids: set[str] | None = None,
    strict_before: bool = True,
) -> ReplayResult:
    """Apply structured patches deterministically to a baseline plan.

    ``patch_ids`` can select the supervised view.  Omitting it replays every
    patch, including weak-but-safe context patches.
    """
    actions_by_index = {
        int(action["source_index"]): deepcopy(action)
        for action in baseline_plan.get("actions", [])
    }
    applied: list[str] = []
    errors: list[dict[str, Any]] = []

    for patch in patches:
        patch_id = str(patch.get("patch_id", ""))
        if patch_ids is not None and patch_id not in patch_ids:
            continue
        patch_type = patch.get("patch_type")
        source_index = int(patch["source_index"])
        before = patch.get("before") or {}
        after = patch.get("after") or {}
        action = actions_by_index.get(source_index)

        if patch_type == "ADD_TECHNIQUE" and action is None:
            # Ornament-only source events have no attack in the route.  Retain
            # them as explicit actions so replay is lossless. A pre-attack
            # ornament may introduce a complete plucked action, so preserve
            # any structural snapshot carried by the patch.
            action = {
                "source_index": source_index,
                "attack": bool(after.get("attack", False)),
                "techniques": [],
                "pre_attack_techniques": [],
            }
            for field in STATE_FIELDS:
                if field in after and field not in {"pre_attack_techniques"}:
                    action[field] = after[field]
            actions_by_index[source_index] = action
        if action is None:
            errors.append(_error(patch, "missing_baseline_action"))
            continue

        if strict_before:
            mismatches = {
                key: {"expected": value, "actual": action.get(key)}
                for key, value in before.items()
                if action.get(key) != value
            }
            if mismatches:
                errors.append(_error(patch, "before_mismatch", fields=mismatches))
                continue

        if patch_type in {
            "CHANGE_MODE", "REPOSITION", "CHANGE_FINGER", "CHANGE_ATTACK",
            "SET_COMPOUND_GESTURE",
        }:
            action.update(after)
        elif patch_type == "SET_JIANZI_TEXT":
            action.update(after)
            text = after.get("jianzi_text")
            if isinstance(text, str) and text:
                action.update(derive_jianzi_semantics(text))
        elif patch_type == "ADD_TECHNIQUE":
            technique = after.get("technique")
            if not technique:
                errors.append(_error(patch, "missing_technique"))
                continue
            techniques = action.setdefault("techniques", [])
            if technique not in techniques:
                techniques.append(technique)
            if after.get("placement") == "pre_attack":
                prefixes = action.setdefault("pre_attack_techniques", [])
                if technique not in prefixes:
                    prefixes.append(technique)
        elif patch_type == "REMOVE_TECHNIQUE":
            technique = before.get("technique") or after.get("technique")
            techniques = action.setdefault("techniques", [])
            if technique not in techniques:
                errors.append(_error(patch, "technique_not_present", technique=technique))
                continue
            techniques.remove(technique)
            prefixes = action.setdefault("pre_attack_techniques", [])
            if technique in prefixes:
                prefixes.remove(technique)
        elif patch_type == "NO_OP":
            pass
        else:
            errors.append(_error(patch, "unknown_patch_type", patch_type=patch_type))
            continue
        applied.append(patch_id)

    return ReplayResult(
        valid=not errors,
        actions=[actions_by_index[index] for index in sorted(actions_by_index)],
        applied_patch_ids=applied,
        errors=errors,
    )


def compare_replay_to_reference(
    actions: Iterable[dict[str, Any]],
    references: Iterable[dict[str, Any]],
    *,
    confidence_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return field-level mismatches for fields asserted by safe references."""
    by_index = {int(action["source_index"]): action for action in actions}
    mismatches: list[dict[str, Any]] = []
    for reference in references:
        if confidence_classes and reference.get("confidence_class") not in confidence_classes:
            continue
        index = int(reference["source_index"])
        action = by_index.get(index)
        if action is None:
            mismatches.append({"source_index": index, "code": "missing_replayed_action"})
            continue
        fields = set(reference.get("explicit_fields", [])) | set(reference.get("inherited_fields", []))
        fields &= set(STATE_FIELDS)
        for field in sorted(fields):
            expected = reference.get(field)
            if expected is not None and action.get(field) != expected:
                mismatches.append({
                    "source_index": index, "code": "reference_mismatch", "field": field,
                    "expected": expected, "actual": action.get(field),
                })
        expected_techniques = set(reference.get("techniques") or [])
        actual_techniques = set(action.get("techniques") or [])
        for technique in sorted(expected_techniques - actual_techniques):
            mismatches.append({
                "source_index": index, "code": "missing_reference_technique",
                "technique": technique,
            })
    return mismatches


def compare_replay_to_patch_targets(
    actions: Iterable[dict[str, Any]], patches: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verify final last-write targets after cumulative corrective batches.

    Later patches may intentionally correct a field written by an earlier
    preview. The final state therefore needs to match the last assertion for
    each field/technique, not every historical intermediate value at once.
    """
    by_index = {int(action["source_index"]): action for action in actions}
    mismatches = []
    field_targets: dict[tuple[int, str], tuple[dict[str, Any], Any]] = {}
    technique_targets: dict[tuple[int, str], tuple[dict[str, Any], bool]] = {}
    placement_targets: dict[tuple[int, str], tuple[dict[str, Any], str]] = {}
    for patch in patches:
        source_index = int(patch["source_index"])
        patch_type = patch["patch_type"]
        if patch_type == "ADD_TECHNIQUE":
            technique = (patch.get("after") or {}).get("technique")
            if technique:
                technique_targets[(source_index, str(technique))] = (patch, True)
                placement = (patch.get("after") or {}).get("placement")
                if placement:
                    placement_targets[(source_index, str(technique))] = (
                        patch, str(placement)
                    )
            for field, expected in (patch.get("after") or {}).items():
                if field not in {"technique", "placement", "pre_attack_techniques"}:
                    field_targets[(source_index, field)] = (patch, expected)
            continue
        if patch_type == "REMOVE_TECHNIQUE":
            technique = ((patch.get("before") or {}).get("technique")
                         or (patch.get("after") or {}).get("technique"))
            if technique:
                technique_targets[(source_index, str(technique))] = (patch, False)
            continue
        for field, expected in (patch.get("after") or {}).items():
            field_targets[(source_index, field)] = (patch, expected)

    for (source_index, field), (patch, expected) in field_targets.items():
        action = by_index.get(source_index)
        if action is None:
            mismatches.append(_error(patch, "missing_replayed_action"))
            continue
        actual = action.get(field)
        # A repeat-omission row may be represented either by the explicit
        # placeholder text or by an empty display cell. Both preserve the
        # underlying sounding event; accept them as equivalent transport
        # spellings while keeping the marker visible in private reference.
        equivalent_repeat_omission = (
            field == "jianzi_text"
            and {actual, expected} <= {"", REPEAT_OMISSION_TEXT}
            and actual != expected
        )
        if actual != expected and not equivalent_repeat_omission:
            mismatches.append(_error(
                patch, "patch_target_mismatch", field=field,
                expected=expected, actual=actual,
            ))
    for (source_index, technique), (patch, should_exist) in technique_targets.items():
        action = by_index.get(source_index)
        if action is None:
            mismatches.append(_error(patch, "missing_replayed_action"))
            continue
        exists = technique in (action.get("techniques") or [])
        if exists != should_exist:
            mismatches.append(_error(
                patch,
                "missing_patch_technique" if should_exist else "unexpected_patch_technique",
                technique=technique,
            ))
    for (source_index, technique), (patch, placement) in placement_targets.items():
        action = by_index.get(source_index)
        if action is None:
            mismatches.append(_error(patch, "missing_replayed_action"))
            continue
        is_pre_attack = technique in (action.get("pre_attack_techniques") or [])
        if placement == "pre_attack" and not is_pre_attack:
            mismatches.append(_error(
                patch, "technique_placement_mismatch", technique=technique,
                expected=placement, actual="suffix",
            ))
    return mismatches
