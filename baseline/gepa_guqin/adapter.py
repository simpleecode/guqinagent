"""GEPAAdapter implementation for the Guqin single-stage agent.

The candidate is a single component, ``{"single_stage": <system prompt>}``;
the seed is the production ``public_system_for("single_stage")`` text.
``evaluate`` runs one deterministic-tool rollout per phrase and scores it
against the sealed reference with :mod:`baseline.gepa_guqin.metric`;
``make_reflective_dataset`` exposes the public prompt, the final jianzi rows
and the deterministic feedback to the reflection model.  Reference notation
text never enters the reflective dataset.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gepa.core.adapter import EvaluationBatch  # noqa: E402

from .compat import action_text  # noqa: E402
from .handbook import compose_system  # noqa: E402
from .metric import DEFAULT_WEIGHTS, _row_order_key, phrase_report  # noqa: E402
from .rollout import GlmBackend, ScoreRunState, run_single_stage_phrase  # noqa: E402

COMPONENT = "single_stage"
# Phrases whose reference contains no evaluable events score this neutral
# constant; it is identical for every candidate, so acceptance comparisons
# are unaffected.
NEUTRAL_SCORE = 0.5


def final_preview_text(session: dict[str, Any]) -> str:
    """Last accepted edit_plan preview text from the rollout trace."""
    last_text = ""
    for round_entry in session.get("trace") or []:
        for result in round_entry.get("tool_results") or []:
            if result.get("name") != "edit_plan":
                continue
            payload = result.get("result") or {}
            if isinstance(payload, dict):
                nested = payload.get("result") if isinstance(payload.get("result"), dict) else payload
                last_text = str(nested.get("text") or "")
    return last_text


def warning_summary(preview_text: str, limit: int = 12) -> str:
    messages = re.findall(r":warning:([^\n|]+)", preview_text)
    seen: list[str] = []
    for message in messages:
        message = message.strip()
        if message and message not in seen:
            seen.append(message)
    if not seen:
        return ""
    return "；".join(seen[:limit]) + (f"（共 {len(seen)} 类）" if len(seen) > limit else "")


class GuqinAgentAdapter:
    # gepa's ReflectiveMutationProposer accesses this attribute directly; it
    # must exist even when unused (None = use the reflection LM proposer).
    propose_new_texts = None

    def __init__(self, examples: list[dict[str, Any]], historical: dict[tuple[str, str], Any],
                 backend: GlmBackend, *, max_rounds: int = 8, attempts: int = 2,
                 sample_temperature: float = 0.25, tolerance_cents: float = 50.0,
                 weights: dict[str, float] | None = None, log: Any = None,
                 with_handbook: bool = True):
        self.examples = examples
        self.examples_by_id = {str(example["sample_id"]): example for example in examples}
        self.historical = historical
        self.backend = backend
        self.max_rounds = max_rounds
        self.attempts = attempts
        self.sample_temperature = sample_temperature
        self.tolerance_cents = tolerance_cents
        self.weights = weights or DEFAULT_WEIGHTS
        self.log = log
        # False = the ablation arm that sees the bare strategy prompt only.
        self.with_handbook = with_handbook
        self.leakage_counts = {"feedback_leaks": 0, "record_soft_hits": 0,
                               "unresolved_trajectory": 0}
        # All phrases of every selected score in source order; prefixes of a
        # requested phrase are rolled out with the *same candidate* so the
        # handoff is always the candidate's own previous output.
        self._score_rows: dict[str, list[dict[str, Any]]] = {}
        for row in self.historical.values():
            self._score_rows.setdefault(str(row["score_key"]), []).append(row)
        for rows in self._score_rows.values():
            rows.sort(key=_row_order_key)
        # sample_id -> session cache per candidate text; bounded to a few
        # candidates (parent + recent proposals) to bound memory.
        self._session_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self.prefix_rollouts = 0

    # -- GEPAAdapter -----------------------------------------------------

    def evaluate(self, batch: list[dict[str, Any]], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        # The handbook is fixed context for every candidate; GEPA mutates only
        # the strategy text in candidate[COMPONENT].
        system_text = (compose_system(candidate[COMPONENT]) if self.with_handbook
                       else candidate[COMPONENT])
        cache_key = hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:16]
        if cache_key not in self._session_cache and len(self._session_cache) >= 4:
            self._session_cache.pop(next(iter(self._session_cache)))
        sessions = self._session_cache.setdefault(cache_key, {})

        requested = {str(example["sample_id"]): example for example in batch}
        by_score: dict[str, list[str]] = {}
        for sid, example in requested.items():
            by_score.setdefault(str(example["score_key"]), []).append(sid)

        for score_key, sample_ids in by_score.items():
            state = ScoreRunState()
            poisoned = False
            score_rows = self._score_rows.get(score_key, [])
            # Roll out only up to the last requested phrase of this score;
            # phrases after it cannot influence any requested handoff.
            last_position = max(
                (position for position, row in enumerate(score_rows)
                 if str(row["trajectory_id"]) in requested),
                default=-1,
            )
            for row in score_rows[:last_position + 1]:
                sample_id = str(row["trajectory_id"])
                working = {
                    "trajectory_id": row["trajectory_id"],
                    "score_key": row["score_key"],
                    "phrase_id": row["phrase_id"],
                    "split": row.get("split"),
                    "input": deepcopy(row["input"]),
                    "baseline_plan": deepcopy(row.get("baseline_plan") or {}),
                }
                injected = state.prepare(working)
                session = sessions.get(sample_id)
                if session is None:
                    if poisoned:
                        # A predecessor failed: later phrases of this score
                        # must not be run against an incomplete history.
                        session = {"ok": False, "trace": [],
                                   "error": "predecessor_phrase_failed"}
                    else:
                        example_view = {
                            "sample_id": sample_id, "score_key": score_key,
                            "phrase_id": working["phrase_id"],
                            "runtime_item": working,
                        }
                        try:
                            session = run_single_stage_phrase(
                                example_view, state.historical, system_text,
                                self.backend, max_rounds=self.max_rounds,
                                attempts=self.attempts,
                                sample_temperature=self.sample_temperature,
                                log=self.log,
                            )
                        except Exception as exc:
                            session = {"ok": False, "trace": [],
                                       "error": f"{type(exc).__name__}: {exc}"}
                        if sample_id not in requested:
                            self.prefix_rollouts += 1
                    sessions[sample_id] = session
                if sample_id in requested and injected is not None:
                    if self.log:
                        self.log({
                            "event": "handoff_provenance", "sample_id": sample_id,
                            "previous_sample_id": self.previous_sample_id(state),
                            "source": "candidate_self_session",
                            "previous_ok": True,
                            "injected_rows": len(injected.get("actions") or []),
                        })
                if not state.commit(working, session):
                    poisoned = True

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        for example in batch:
            session = sessions.get(str(example["sample_id"])) or {"ok": False}
            trajectory, output, score, objective = self._score_one(example, session)
            outputs.append(output)
            scores.append(score)
            objective_scores.append(objective)
            if capture_traces:
                assert trajectories is not None
                trajectories.append(trajectory)
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories,
            objective_scores=objective_scores,
        )

    @staticmethod
    def previous_sample_id(state: ScoreRunState) -> str | None:
        previous = state.previous
        return str(previous.get("trajectory_id")) if previous else None

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        dataset: dict[str, list[Mapping[str, Any]]] = {}
        for component in components_to_update:
            records: list[Mapping[str, Any]] = []
            for trajectory in eval_batch.trajectories or []:
                record = {
                    "Inputs": {"user_prompt": trajectory["user_prompt"]},
                    "Generated Outputs": {
                        "jianzi_rows": trajectory["jianzi_rows"],
                        "final_preview": trajectory["final_preview_text"],
                    },
                    "Feedback": trajectory["feedback"],
                    "score": trajectory["score"],
                }
                self._audit_leakage(component, record, trajectory)
                records.append(record)
            dataset[component] = records
        return dataset

    def _audit_leakage(self, component: str, record: Mapping[str, Any],
                       trajectory: Mapping[str, Any]) -> None:
        """Verify no *current-phrase* reference notation reached reflection.

        The read-only previous-phrase handoff inside ``user_prompt`` is public
        protocol (mirrors the production loop's own-history handoff) and is
        deliberately out of scope.  Anything else containing the current
        phrase's reference jianzi is a leak: a Feedback hit is a hard failure
        (feedback is generated from deterministic counts only); Inputs/Outputs
        hits are reported for manual review because the model may legitimately
        reproduce reference notation on its own.
        """
        example = self.examples_by_id.get(str(trajectory.get("sample_id")))
        if example is None:
            self.leakage_counts["unresolved_trajectory"] += 1
            return
        reference_texts = {
            action_text(action).strip()
            for action in example["reference"]["actions"]
            if len(action_text(action).strip()) >= 4
        }
        if not reference_texts:
            return
        feedback = str(record.get("Feedback") or "")
        leaked = sorted(text for text in reference_texts if text in feedback)
        if leaked:
            self.leakage_counts["feedback_leaks"] += len(leaked)
            if self.log:
                self.log({"event": "leakage_feedback", "component": component,
                          "sample_id": example["sample_id"], "leaked": leaked[:3]})
        serialized = json.dumps(record, ensure_ascii=False, default=str)
        self.leakage_counts["record_soft_hits"] += sum(
            1 for text in reference_texts if text in serialized)

    # -- internals ---------------------------------------------------------

    def _score_one(self, example: dict[str, Any],
                   session: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, float]]:
        ok = bool(session.get("ok"))
        actions = ((session.get("plan") or {}).get("actions")) or [] if ok else []
        jianzi_rows = [[int(action["source_index"]), action_text(action)]
                       for action in actions if action.get("source_index") is not None]
        preview = final_preview_text(session)
        prediction = {"score_key": example["score_key"],
                      "phrase_id": example["phrase_id"],
                      "jianzi_rows": jianzi_rows}
        if ok:
            report = phrase_report(
                example, prediction, self.historical,
                tolerance_cents=self.tolerance_cents, weights=self.weights,
                warnings_text=warning_summary(preview),
            )
            if report["score"] is None:
                score = NEUTRAL_SCORE
                report["feedback"] = ("本段没有可评估的演奏事件；按中性分计。"
                                      + ("" if ok else ""))
            else:
                score = float(report["score"])
        else:
            score = 0.0
            rounds = len(session.get("trace") or [])
            errors = [
                str(entry.get("generation_error"))
                for entry in session.get("trace") or []
                if entry.get("generation_error")
            ]
            failure = session.get("error") or (errors[-1] if errors else
                                               f"{rounds} 轮内未获得有效 edit_plan 预览")
            report = {
                "score": 0.0,
                "objective_scores": {key: 0.0 for key in
                                     ("pitch_accuracy", "tone_type_accuracy",
                                      "left_hand_f1", "right_hand_f1",
                                      "ornament_f1", "playability")},
                "feedback": f"协议失败（得分 0）：{failure}",
                "metrics": {"events": 0, "protocol_ok": False},
            }
        if self.log:
            self.log({"event": "phrase_scored", "sample_id": example.get("sample_id"),
                      "ok": ok, "score": score,
                      "usage": self.backend.client.snapshot(),
                      "leakage": dict(self.leakage_counts)})
        trajectory = {
            "sample_id": example.get("sample_id"),
            "user_prompt": session.get("user_prompt") or "",
            "jianzi_rows": jianzi_rows,
            "final_preview_text": preview,
            "feedback": report["feedback"],
            "score": score,
            "protocol_ok": ok,
            "rounds": len(session.get("trace") or []),
            "trace": session.get("trace") or [],
        }
        output = {
            "sample_id": example.get("sample_id"),
            "protocol_ok": ok,
            "jianzi_rows": jianzi_rows,
            "final_preview_text": preview,
        }
        return trajectory, output, score, dict(report["objective_scores"])
