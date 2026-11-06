"""GEPA adapter using the senior exact-cell scorer and label-derived feedback.

The actor never receives references. Reflection can receive either label features
or exact train-only generated/reference text differences. Validation/test labels
are never included in reflection. Candidate instructions may not contain dataset IDs.
"""
from __future__ import annotations

import hashlib
import difflib
import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch

from baseline.scripts.teacher_standard_score import canonical, score_prediction

from .adapter import GuqinAgentAdapter, final_preview_text
from .compat import action_text


RIGHT = ("抹", "挑", "勾", "剔", "擘", "托", "打", "摘")
LEFT = ("大指", "名指", "中指", "食指")
TECHNIQUES = ("吟", "猱", "撞", "绰", "注", "上", "下", "复", "就", "历", "滚", "拂", "撮", "打圆", "再作")


def _mode(text: str) -> str:
    value = canonical(text)
    if "泛" in value:
        return "泛"
    if "散" in value:
        return "散"
    if "按" in value or "徽" in value:
        return "按"
    return "继承/未明"


def _features(text: str) -> dict[str, Any]:
    value = canonical(text)
    return {
        "按散泛": _mode(value),
        "弦序": re.findall(r"([1-7])弦", value),
        "是否写徽位": "徽" in value,
        "左手": [token for token in LEFT if token in value],
        "右手": [token for token in RIGHT if token in value],
        "技法": [token for token in TECHNIQUES if token in value],
        "规范化字数": len(value),
    }


def _difference(index: int, reference: str, prediction: str) -> tuple[str, str]:
    ref, pred = canonical(reference), canonical(prediction)
    if ref == pred:
        return "exact", f"音序 {index}：与标注完全一致"
    if ref and not pred:
        return "missing", f"音序 {index}：漏填；标注特征={json.dumps(_features(ref), ensure_ascii=False)}"
    if pred and not ref:
        return "extra", f"音序 {index}：多填；该格标注为空，生成特征={json.dumps(_features(pred), ensure_ascii=False)}"
    ref_features, pred_features = _features(ref), _features(pred)
    if ref_features["按散泛"] != pred_features["按散泛"]:
        kind = "mode"
    elif ref_features["弦序"] != pred_features["弦序"]:
        kind = "string"
    elif ref_features["是否写徽位"] != pred_features["是否写徽位"]:
        kind = "hui"
    else:
        kind = "notation"
    return kind, (
        f"音序 {index}：{kind} 差异；"
        f"标注特征={json.dumps(ref_features, ensure_ascii=False)}；"
        f"生成特征={json.dumps(pred_features, ensure_ascii=False)}"
    )


def teacher_feedback(reference_actions: list[dict[str, Any]], prediction: dict[str, Any], *,
                     protocol_ok: bool, failure: str | None = None,
                     max_details: int = 24,
                     include_reference: bool = False) -> tuple[dict[str, Any], str, dict[str, int]]:
    if not protocol_ok:
        score = score_prediction(reference_actions, prediction)
        return score, f"协议失败：{failure or 'no_valid_edit_plan'}。本样本逐格精确率按0计。", {"protocol_failure": 1}

    score = score_prediction(reference_actions, prediction)
    reference = {
        int(item["source_index"]): str(item.get("text") or item.get("jianzi_text") or "")
        for item in reference_actions if item.get("source_index") is not None
    }
    generated = {int(item[0]): str(item[1]) for item in prediction.get("jianzi_rows") or []}
    counts: Counter[str] = Counter()
    details: list[str] = []
    if include_reference:
        max_details = len(set(reference) | set(generated))
    for index in sorted(set(reference) | set(generated)):
        kind, message = _difference(index, reference.get(index, ""), generated.get(index, ""))
        counts[kind] += 1
        if kind != "exact" and len(details) < max_details:
            if include_reference:
                ref, pred = reference.get(index, ""), generated.get(index, "")
                edits = []
                for tag, a, b, c, d in difflib.SequenceMatcher(None, canonical(pred), canonical(ref)).get_opcodes():
                    if tag != "equal":
                        edits.append(f"{tag}: {canonical(pred)[a:b]!r}→{canonical(ref)[c:d]!r}")
                message += (f"；实际生成={json.dumps(pred, ensure_ascii=False)}"
                            f"；训练标注={json.dumps(ref, ensure_ascii=False)}；规范化文字差分=" + "、".join(edits))
            details.append(message)
    omitted = sum(counts.values()) - counts["exact"] - len(details)
    exact_rate = score["exact_cell_rate"]
    nonempty_recall = score["nonempty_recall"]
    header = (
        f"师兄标准：逐格精确率 {f'{exact_rate:.3f}' if exact_rate is not None else 'N/A'}；"
        f"非空召回 {f'{nonempty_recall:.3f}' if nonempty_recall is not None else 'N/A'}；协议合规。"
        f"差异统计：{json.dumps(dict(counts), ensure_ascii=False)}。"
    )
    suffix = f"其余 {omitted} 个差异未展开。" if omitted > 0 else ""
    feedback = header + ("逐格反馈：" + "；".join(details) if details else "所有格均一致。") + suffix
    return score, feedback, dict(counts)


class TeacherExactGuqinAdapter(GuqinAgentAdapter):
    """Guqin adapter whose GEPA scalar is exact-cell rate.

    Protocol failures score zero.  Recall and protocol are retained as objective
    diagnostics and are enforced again by the post-run safety gate.
    """

    def __init__(self, *args: Any, feedback_detail: str = "features", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.feedback_detail = feedback_detail
        self.leakage_counts["train_label_feedback_records"] = 0
        self._active_candidate_sha256 = ""
        self._feedback_batch = 0
        self.audit_context: dict[str, Any] = {}

    def evaluate(self, batch: list[dict[str, Any]], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        self._active_candidate_sha256 = hashlib.sha256(
            candidate["single_stage"].encode("utf-8")
        ).hexdigest()
        text = candidate["single_stage"]
        identifiers = {str(example[key]) for example in getattr(self, "examples", [])
                       for key in ("sample_id", "score_key") if example.get(key)}
        if any(identifier in text for identifier in identifiers):
            scored = [self._score_one(example, {"ok": False, "trace": [],
                      "error": "candidate_contains_dataset_identifiers; use general rules instead of memorized IDs"})
                      for example in batch]
            return EvaluationBatch(outputs=[row[1] for row in scored], scores=[row[2] for row in scored],
                                   trajectories=[row[0] for row in scored] if capture_traces else None,
                                   objective_scores=[row[3] for row in scored])
        result = super().evaluate(batch, candidate, capture_traces=capture_traces)
        # A provider/network outage must abort the trial, rather than silently
        # becoming a low prompt score. Also inspect failed prefix sessions.
        for sessions in self._session_cache.values():
            for session in sessions.values():
                if not session.get("ok"):
                    errors = [str(session.get("error") or "")]
                    errors.extend(str(entry.get("generation_error") or "") for entry in session.get("trace") or [])
                    if any(re.search(r"APITimeoutError|APIConnectionError|RateLimitError|AuthenticationError|PermissionDeniedError|InternalServerError", error)
                           for error in errors):
                        raise RuntimeError("GLM基础设施调用失败；本次评测不用于筛选提示词。检查接口/网络后执行原命令恢复。")
        return result

    def _score_one(self, example: dict[str, Any], session: dict[str, Any]):
        ok = bool(session.get("ok"))
        actions = (((session.get("plan") or {}).get("actions")) or []) if ok else []
        rows = [[int(action["source_index"]), action_text(action)]
                for action in actions if action.get("source_index") is not None]
        prediction = {
            "score_key": example["score_key"], "phrase_id": example["phrase_id"],
            "jianzi_rows": rows,
        }
        failure = None if ok else str(session.get("error") or "no_valid_edit_plan")
        teacher_score, feedback, differences = teacher_feedback(
            example["reference"]["actions"], prediction,
            protocol_ok=ok, failure=failure,
        )
        feedback_without_labels = feedback
        labeled_feedback = (self.feedback_detail == "exact" and
                            example.get("split") in {"feedback", "D_feedback", "gepa_train"})
        if labeled_feedback:
            _, feedback, _ = teacher_feedback(example["reference"]["actions"], prediction,
                                              protocol_ok=ok, failure=failure, include_reference=True)
        scalar = float(teacher_score.get("exact_cell_rate") or 0.0) if ok else 0.0
        objective = {
            "exact_cell_rate": scalar,
            "nonempty_recall": float(teacher_score.get("nonempty_recall") or 0.0),
            "protocol_valid_rate": 1.0 if ok else 0.0,
        }
        preview = final_preview_text(session)
        event = {
            "event": "teacher_phrase_scored", "sample_id": example.get("sample_id"),
            "candidate_sha256": self._active_candidate_sha256, "ok": ok,
            "score": scalar, "teacher_score": teacher_score,
            "differences": differences, "error": failure,
        }
        if self.log:
            self.log({**event, **self.audit_context, "prediction": prediction})
        trajectory = {
            "sample_id": example.get("sample_id"),
            "candidate_sha256": self._active_candidate_sha256,
            "user_prompt": session.get("user_prompt") or "",
            "jianzi_rows": rows,
            "final_preview_text": preview,
            "feedback": feedback,
            "feedback_without_labels": feedback_without_labels,
            "feedback_access": "training_labels" if labeled_feedback else "features_only",
            "difference_counts": differences,
            "teacher_score": teacher_score,
            "score": scalar,
            "protocol_ok": ok,
            "rounds": len(session.get("trace") or []),
            "trace": session.get("trace") or [],
        }
        output = {
            "sample_id": example.get("sample_id"), "protocol_ok": ok,
            "jianzi_rows": rows, "final_preview_text": preview,
            "teacher_score": teacher_score, "difference_counts": differences,
        }
        return trajectory, output, scalar, objective

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        self._feedback_batch += 1
        result: dict[str, list[Mapping[str, Any]]] = {}
        for component in components_to_update:
            records: list[Mapping[str, Any]] = []
            for trajectory in eval_batch.trajectories or []:
                record = {
                    "sample_id": trajectory.get("sample_id"),
                    "Inputs": {"user_prompt": trajectory["user_prompt"]},
                    "Generated Outputs": {"jianzi_rows": trajectory["jianzi_rows"]},
                    "Tool execution": [
                        {
                            "round": entry.get("round"),
                            "tool_calls": entry.get("tool_calls") or [],
                            "tool_results_excerpt": json.dumps(
                                entry.get("tool_results") or [], ensure_ascii=False,
                            )[:1800],
                            "error": entry.get("generation_error"),
                        }
                        for entry in trajectory.get("trace") or []
                    ],
                    "Feedback": trajectory["feedback"],
                    "Feedback access": trajectory.get("feedback_access", "features_only"),
                    "Difference counts": trajectory.get("difference_counts", {}),
                    "score": trajectory["score"],
                    "Previous trial evidence": getattr(self, "evolution_memory", [])[-6:],
                    "instruction": (
                        "归纳可泛化规则，只修改导致这类差异的指令。"
                        "不得在新提示词中写入 sample_id、score_key、特定音序的标准答案或整段标注。"
                    ),
                }
                # Explicit train-label access is intentional. Audit the
                # answer-free channel separately; actor input remains clean.
                audit_record = dict(record)
                audit_record["Feedback"] = trajectory.get("feedback_without_labels", trajectory["feedback"])
                self._audit_leakage(component, audit_record, trajectory)
                if trajectory.get("feedback_access") == "training_labels":
                    self.leakage_counts["train_label_feedback_records"] += 1
                records.append(record)
                if self.log:
                    self.log({
                        "event": "reflection_feedback",
                        **self.audit_context,
                        "feedback_batch": self._feedback_batch,
                        "component": component,
                        "sample_id": trajectory.get("sample_id"),
                        "candidate_sha256": trajectory.get("candidate_sha256"),
                        "score": trajectory.get("score"),
                        "feedback": trajectory.get("feedback"),
                        "feedback_access": trajectory.get("feedback_access", "features_only"),
                        "difference_counts": trajectory.get("difference_counts"),
                    })
            result[component] = records
        return result
