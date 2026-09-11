from __future__ import annotations

from copy import deepcopy
from typing import Any


EDITABLE_FIELDS = {
    "mode", "string", "hui", "left_finger", "right_finger", "attack",
    "attack_source_id", "techniques",
    "compound_gesture", "pre_attack_techniques",
}


def apply_plan_edits(plan: dict[str, Any], *, current_revision: int,
                     base_revision: int, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and apply local edits without mutating the input plan."""
    if base_revision != current_revision:
        return {"ok": False, "revision": current_revision, "plan": plan,
                "errors": [{"code": "STALE_REVISION", "expected": current_revision,
                            "actual": base_revision}]}
    updated = deepcopy(plan)
    by_id = {action["action_id"]: action for action in updated.get("actions", [])}
    errors, applied = [], []
    for number, operation in enumerate(operations, 1):
        action_id = operation.get("action_id")
        action = by_id.get(action_id)
        if action is None:
            errors.append({"operation": number, "code": "UNKNOWN_ACTION", "action_id": action_id})
            continue
        expected = operation.get("expected") or {}
        mismatch = {field: {"expected": value, "actual": action.get(field)}
                    for field, value in expected.items() if action.get(field) != value}
        if mismatch:
            errors.append({"operation": number, "code": "PRECONDITION_FAILED",
                           "action_id": action_id, "fields": mismatch})
            continue
        op = operation.get("op")
        if op == "set_fields":
            fields = operation.get("fields") or {}
            illegal = set(fields) - EDITABLE_FIELDS
            if illegal:
                errors.append({"operation": number, "code": "ILLEGAL_FIELDS",
                               "fields": sorted(illegal)})
                continue
            action.update(fields)
        elif op == "add_technique":
            technique = str(operation.get("technique") or "").strip()
            if not technique:
                errors.append({"operation": number, "code": "EMPTY_TECHNIQUE"})
                continue
            techniques = action.setdefault("techniques", [])
            if technique not in techniques:
                techniques.append(technique)
        elif op == "remove_technique":
            technique = str(operation.get("technique") or "").strip()
            techniques = action.setdefault("techniques", [])
            if technique not in techniques:
                errors.append({"operation": number, "code": "TECHNIQUE_NOT_PRESENT",
                               "technique": technique})
                continue
            techniques.remove(technique)
        else:
            errors.append({"operation": number, "code": "UNKNOWN_OPERATION", "op": op})
            continue
        applied.append(number)
    if errors:
        return {"ok": False, "revision": current_revision, "plan": plan,
                "applied_operations": [], "errors": errors}
    return {"ok": True, "revision": current_revision + 1, "plan": updated,
            "applied_operations": applied, "errors": []}
