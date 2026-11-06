"""Multi-objective retention for prompt experiments; it never alters benchmark scoring."""
from __future__ import annotations
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_rows(rows: list[dict[str, Any]], arms: list[str], weights: dict[str, float]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok_rows = [row for row in arm_rows if row.get("protocol_ok")]
        component_means, component_coverage = {}, {}
        for component in weights:
            values = [float(row["components"][component]) for row in ok_rows if row.get("components", {}).get(component) is not None]
            component_means[component], component_coverage[component] = _mean(values), len(values)
        summary[arm] = {"n": len(arm_rows), "protocol_successes": len(ok_rows),
                        "mean_score": _mean([float(row.get("score", 0.0)) for row in arm_rows]),
                        "total_rule_violations": sum(int(row.get("rule_violations") or 0) for row in arm_rows),
                        "component_means": component_means, "component_coverage": component_coverage,
                        "mean_actor_calls": _mean([float(row.get("actor_calls", 0)) for row in arm_rows])}
    return summary


def pareto_decisions(summary: dict[str, dict[str, Any]], baseline: str, weights: dict[str, float], eps: float = 1e-9) -> dict[str, dict[str, Any]]:
    parent = summary.get(baseline)
    if not parent:
        return {}
    outcomes: dict[str, dict[str, Any]] = {}
    for name, candidate in summary.items():
        if name == baseline:
            continue
        p0, p1 = parent["component_means"].get("pitch"), candidate["component_means"].get("pitch")
        gates = {"protocol": candidate["protocol_successes"] >= parent["protocol_successes"],
                 "pitch": p0 is None or (p1 is not None and p1 + eps >= p0),
                 "violations": candidate["total_rule_violations"] <= parent["total_rule_violations"]}
        safe = all(gates.values())
        gains = [component for component in weights if candidate["component_coverage"].get(component) == parent["component_coverage"].get(component) and candidate["component_means"].get(component) is not None and parent["component_means"].get(component) is not None and candidate["component_means"][component] > parent["component_means"][component] + eps]
        total_better = candidate["mean_score"] is not None and parent["mean_score"] is not None and candidate["mean_score"] > parent["mean_score"] + eps
        outcomes[name] = {"safe_gate": safe, "gate_detail": gates,
                          "component_specialist_for": gains if safe else [],
                          "global_parent_eligible": bool(safe and total_better),
                          "retain_as_evidence": bool(safe and (total_better or gains)),
                          "rule": "只提取明确规则；不可整段复活低总分专家提示词。"}
    return outcomes
