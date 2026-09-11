from __future__ import annotations

import platform
import copy
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .artifacts import ArtifactStore
from .backends import AgentBackend
from .context_handoff import render_context_source
from .pitch_candidates import _pitch
from .plan_editor import apply_plan_edits
from .plan_source import render_diagnostics, render_plan_source
from .role_agents import RoleAgentFactory
from .serialization import (
    candidate_from_dict,
    plan_from_dict,
    route_from_dict,
    score_from_dict,
)
from .skills import SkillRegistry, default_skill_registry


DEFAULT_CONFIG: dict[str, Any] = {
    "tuning_strategy": "auto",
    "performance_context": "solo",
    "open_tone_demand": "auto",
    "tuning_name": "正调",
    "open_midi": [48, 50, 53, 55, 57, 60, 62],
    "phrase_bars": 8,
    "candidate_tolerance_cents": 35.0,
    "audit_tolerance_cents": 35.0,
    "candidate_modes": ["stopped"],
    "max_candidates": 18,
    "top_k": 5,
    "states_per_candidate": 3,
    "max_repairs": 1,
}


def _merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    job_id: str
    abc_text: str
    config: dict[str, Any]
    score: dict[str, Any]
    tuning_decision: dict[str, Any]
    analysis: dict[str, Any]
    candidates: dict[str, list[dict[str, Any]]]
    phrase_handoffs: dict[str, dict[str, Any]]
    phrase_id: str
    phrase_handoff: dict[str, Any]
    phrase_routes: Annotated[dict[str, list[dict[str, Any]]], _merge_dict]
    selected_routes: list[dict[str, Any]]
    route_merge: dict[str, Any]
    performance_plan: dict[str, Any]
    plan_revision: int
    plan_revisions: list[dict[str, Any]]
    plan_edits: list[dict[str, Any]]
    fingering_optimization: dict[str, Any]
    compiled: dict[str, Any]
    style_review: dict[str, Any]
    audit: dict[str, Any]
    repair_count: int
    status: str


class AbcToJianzipuOrchestrator:
    """LangGraph coordinator around deterministic, versioned music skills.

    Phrase route planning is dispatched as a map/reduce subagent stage.  The
    current MVP uses deterministic handlers; model-backed analyst, style, and
    repair agents can be inserted without changing the score or plan schemas.
    """

    def __init__(self, artifact_root: str | Path = "runs/abc_to_jianzipu", *,
                 registry: SkillRegistry | None = None,
                 backend: AgentBackend | None = None,
                 checkpointer: Any | None = None) -> None:
        self.artifact_root = Path(artifact_root)
        self.registry = registry or default_skill_registry()
        self.agent_factory = RoleAgentFactory(self.registry, backend)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("parse_score", self._parse_score)
        builder.add_node("plan_tuning", self._plan_tuning)
        builder.add_node("analyze_score", self._analyze_score)
        builder.add_node("generate_candidates", self._generate_candidates)
        builder.add_node("plan_phrase", self._plan_phrase)
        builder.add_node("merge_routes", self._merge_routes)
        builder.add_node("build_plan", self._build_plan)
        builder.add_node("optimize_fingering", self._optimize_fingering)
        builder.add_node("compile", self._compile)
        builder.add_node("style_review", self._style_review)
        builder.add_node("audit", self._audit)
        builder.add_node("repair", self._repair)

        builder.add_edge(START, "parse_score")
        builder.add_edge("parse_score", "plan_tuning")
        builder.add_edge("plan_tuning", "analyze_score")
        builder.add_edge("analyze_score", "generate_candidates")
        builder.add_conditional_edges(
            "generate_candidates", self._dispatch_phrases, ["plan_phrase"]
        )
        builder.add_edge("plan_phrase", "merge_routes")
        builder.add_edge("merge_routes", "build_plan")
        builder.add_edge("build_plan", "optimize_fingering")
        builder.add_edge("optimize_fingering", "compile")
        builder.add_edge("compile", "style_review")
        builder.add_edge("style_review", "audit")
        builder.add_conditional_edges(
            "audit", self._after_audit, {"done": END, "repair": "repair"}
        )
        builder.add_edge("repair", "generate_candidates")
        return builder.compile(checkpointer=self.checkpointer)

    def _parse_score(self, state: GraphState) -> dict:
        config = state["config"]
        score = self.registry.call(
            "abc.parse", state["abc_text"],
            tuning_name=config["tuning_name"],
            open_midi=config["open_midi"],
            phrase_bars=int(config["phrase_bars"]),
        )
        if not score.events:
            raise ValueError("ABC contains no timed score events")
        return {"score": score.to_dict(), "status": "parsed"}

    def _analyze_score(self, state: GraphState) -> dict:
        result = self.agent_factory.create("score_analyst").run(
            state["job_id"], {"canonical_score": state["score"]},
            skill_versions=self.registry.versions(),
        )
        return {"analysis": result.to_dict()}

    def _plan_tuning(self, state: GraphState) -> dict:
        config = state["config"]
        score = score_from_dict(state["score"])
        if config.get("tuning_strategy", "auto") == "fixed":
            decision = {
                "strategy": "fixed", "selected_name": score.tuning_name,
                "open_midi": score.open_midi, "reason_codes": ["USER_LOCKED_TUNING"],
            }
        else:
            assessment = self.registry.call(
                "guqin.decide_tuning", score,
                performance_context=config.get("performance_context", "solo"),
                open_tone_demand=config.get("open_tone_demand", "auto"),
            )
            ranked = assessment["ranked_candidates"]
            review = self.agent_factory.create("tuning_planner").run(
                state["job_id"],
                {"canonical_score": state["score"], "retuning_assessment": assessment,
                 "performance_context": config.get("performance_context", "solo"),
                 "open_tone_demand": config.get("open_tone_demand", "auto"),
                 "_required_tool": "assess_retuning"},
                skill_versions=self.registry.versions(),
            )
            proposed = review.output if review.status in {"ok", "completed"} else {}
            selected = next((item for item in ranked if (
                item["name"] == proposed.get("selected_name")
                and item["open_midi"] == proposed.get("open_midi")
            )), next(item for item in ranked if item["name"] == assessment["selected_name"]))
            decision = {
                "strategy": "auto", "selected_name": selected["name"],
                "open_midi": selected["open_midi"],
                "retune": selected["name"] != "正调",
                "reason_codes": proposed.get("reason_codes") or assessment["reason_codes"],
                "retuning_assessment": assessment, "agent_review": review.to_dict(),
            }
            score.tuning_name = selected["name"]
            score.open_midi = selected["open_midi"]
        return {"score": score.to_dict(), "tuning_decision": decision, "status": "tuning_planned"}

    def _generate_candidates(self, state: GraphState) -> dict:
        score = score_from_dict(state["score"])
        config = state["config"]
        candidates = self.registry.call(
            "guqin.pitch_candidates", score,
            tolerance_cents=float(config["candidate_tolerance_cents"]),
            modes=tuple(config["candidate_modes"]),
            max_candidates=int(config["max_candidates"]),
        )
        event_by_id = {event.id: event for event in score.events}
        phrase_handoffs = {}
        for phrase in score.phrases:
            current_phrase = [asdict(event_by_id[event_id]) for event_id in phrase.event_ids]
            phrase_handoffs[phrase.id] = {
                "schema_version": "phrase-handoff-runtime-2.0",
                "phrase_id": phrase.id, "section_id": phrase.section_id,
                "current_phrase": current_phrase,
                "context_source": render_context_source(current_phrase),
                "constraints": {
                    "previous_phrase_readonly": True,
                    "only_current_phrase_is_editable": True,
                },
            }
        return {
            "candidates": {
                event_id: [candidate.to_dict() for candidate in items]
                for event_id, items in candidates.items()
            },
            "phrase_handoffs": phrase_handoffs,
            "status": "candidates_generated",
        }

    @staticmethod
    def _dispatch_phrases(state: GraphState) -> list[Send]:
        score = score_from_dict(state["score"])
        return [
            Send("plan_phrase", {
                "job_id": state["job_id"],
                "config": state["config"],
                "score": state["score"],
                "candidates": state["candidates"],
                "phrase_handoff": state["phrase_handoffs"][phrase.id],
                "phrase_id": phrase.id,
            })
            for phrase in score.phrases
        ]

    def _plan_phrase(self, state: GraphState) -> dict:
        score = score_from_dict(state["score"])
        candidates = {
            event_id: [candidate_from_dict(item) for item in items]
            for event_id, items in state["candidates"].items()
        }
        config = state["config"]
        phrase_id = state["phrase_id"]
        phrase = next(item for item in score.phrases if item.id == phrase_id)
        phrase_event_ids = set(phrase.event_ids)
        candidates = {event_id: items for event_id, items in candidates.items()
                      if event_id in phrase_event_ids}
        routes = self.registry.call(
            "guqin.phrase_routes", score, phrase_id, candidates,
            top_k=int(config["top_k"]),
            states_per_candidate=int(config["states_per_candidate"]),
        )
        return {"phrase_routes": {phrase_id: [route.to_dict() for route in routes]}}

    def _merge_routes(self, state: GraphState) -> dict:
        score = score_from_dict(state["score"])
        candidates = {
            event_id: [candidate_from_dict(item) for item in items]
            for event_id, items in state["candidates"].items()
        }
        candidate_index = {
            candidate.candidate_id: candidate
            for items in candidates.values() for candidate in items
        }
        routes_by_phrase = {
            phrase_id: [route_from_dict(item) for item in items]
            for phrase_id, items in state.get("phrase_routes", {}).items()
        }
        merged = self.registry.call(
            "guqin.merge_routes_report", score, routes_by_phrase, candidate_index
        )
        handoffs = copy.deepcopy(state.get("phrase_handoffs", {}))
        previous_route = None
        for route_index, route in enumerate(merged.selected_routes):
            handoff = handoffs.get(route.phrase_id)
            if handoff is not None:
                previous_phrase = None
                older_refs = []
                if previous_route is not None:
                    previous_phrase = {
                        "phrase_id": previous_route.phrase_id,
                        "status": "route_confirmed_fingering_pending",
                        "actions": [
                            {
                                **candidate_index[candidate_id].to_dict(),
                                "source_event_ids": [candidate_index[candidate_id].event_id],
                                "left_finger": None, "right_finger": None,
                                "attack": True, "techniques": [],
                            }
                            for candidate_id in previous_route.candidate_ids
                        ],
                    }
                    handoff["previous_phrase"] = previous_phrase
                if route_index > 1:
                    older_refs = [
                        {
                            "phrase_id": older.phrase_id,
                            "context_ref": (
                                f"context://score/{state['job_id']}/"
                                f"phrases/{older.phrase_id}"
                            ),
                        }
                        for older in merged.selected_routes[:route_index - 1]
                    ]
                    handoff["older_context_refs"] = older_refs
                handoff["context_source"] = render_context_source(
                    handoff["current_phrase"], previous_phrase=previous_phrase,
                    older_context_refs=older_refs,
                )
            previous_route = route
        return {
            "selected_routes": [route.to_dict() for route in merged.selected_routes],
            "route_merge": merged.to_dict(),
            "phrase_handoffs": handoffs,
        }

    def _build_plan(self, state: GraphState) -> dict:
        candidate_index = {
            item["candidate_id"]: candidate_from_dict(item)
            for items in state["candidates"].values() for item in items
        }
        selected = [route_from_dict(item) for item in state["selected_routes"]]
        plan = self.registry.call("guqin.routes_to_plan", selected, candidate_index)
        plan_dict = plan.to_dict()
        merge_report = state.get("route_merge", {})
        if "total_cost" in merge_report:
            plan_dict["total_cost"] = merge_report["total_cost"]
        return {
            "performance_plan": plan_dict, "plan_revision": 0,
            "plan_revisions": [{"revision": 0, "stage": "baseline", "plan": plan_dict}],
            "plan_edits": [], "status": "plan_built",
        }

    def _optimize_fingering(self, state: GraphState) -> dict:
        result = self.agent_factory.create("fingering_agent").run(
            state["job_id"], {
                "canonical_score": state["score"],
                "performance_plan": state["performance_plan"],
                "plan_revision": state.get("plan_revision", 0),
            },
            skill_versions=self.registry.versions(),
        )
        plan = copy.deepcopy(state["performance_plan"])
        action_by_id = {action["action_id"]: action for action in plan.get("actions", [])}
        event_by_id = {event["id"]: event for event in state["score"].get("events", [])}
        open_midi = state["score"]["open_midi"]
        tolerance = float(state["config"]["audit_tolerance_cents"])
        application: list[dict[str, Any]] = []
        operations: list[dict[str, Any]] = []
        replacements = result.output.get("replacements", [])
        if not isinstance(replacements, list):
            replacements = []
            application.append({"status": "rejected", "reason": "replacements_not_a_list"})
        for replacement in replacements:
            if not isinstance(replacement, dict):
                application.append({"status": "rejected", "reason": "replacement_not_an_object"})
                continue
            action_id = replacement.get("action_id")
            action = action_by_id.get(action_id)
            if action is None:
                application.append({
                    "status": "rejected", "action_id": action_id,
                    "reason": "unknown_action_id",
                })
                continue
            try:
                mode = str(replacement.get("mode", action.get("mode") or "stopped"))
                string = int(replacement.get("string", action.get("string")))
                hui_value = replacement.get("hui", action.get("hui"))
                hui = float(hui_value) if hui_value is not None else None
                if mode not in {"stopped", "open", "harmonic"} or not 1 <= string <= 7:
                    raise ValueError("invalid mode or string")
                source_id = action["source_event_ids"][0]
                target = event_by_id[source_id]["pitches_midi"]
                if not target:
                    raise ValueError("source event has no target pitch")
                sounding = _pitch(float(open_midi[string - 1]), hui, mode)
                cents_error = (sounding - max(target)) * 100
                if abs(cents_error) > tolerance:
                    raise ValueError(
                        f"pitch error {cents_error:.3f}c exceeds {tolerance:.3f}c"
                    )
                fields = {"mode": mode, "string": string, "hui": hui}
                for field in ("left_finger", "right_finger"):
                    if field in replacement:
                        fields[field] = replacement[field]
                operations.append({
                    "op": "set_fields", "action_id": action_id,
                    "expected": {key: action.get(key) for key in ("mode", "string", "hui")},
                    "fields": fields, "reason": replacement.get("reason", ""),
                })
                application.append({
                    "status": "validated", "action_id": action_id,
                    "cents_error": round(cents_error, 3),
                    "reason": replacement.get("reason", ""),
                })
            except (KeyError, TypeError, ValueError) as exc:
                application.append({
                    "status": "rejected", "action_id": action_id,
                    "reason": str(exc),
                })
        optimization = result.to_dict()
        edit_envelope = {
            "base_revision": state.get("plan_revision", 0),
            "agent_role": "fingering_agent", "operations": operations,
        }
        edit_result = apply_plan_edits(
            state["performance_plan"], current_revision=state.get("plan_revision", 0),
            base_revision=edit_envelope["base_revision"], operations=operations,
        )
        if edit_result["ok"]:
            plan = edit_result["plan"]
            for item in application:
                if item.get("status") == "validated":
                    item["status"] = "applied"
            for action in plan.get("actions", []):
                if any(op["action_id"] == action["action_id"] for op in operations):
                    action.setdefault("rule_evidence", []).append("AGENT_FINGERING_OPTIMIZATION")
        optimization["application"] = application
        optimization["edit_envelope"] = edit_envelope
        optimization["edit_result"] = {key: value for key, value in edit_result.items() if key != "plan"}
        revisions = list(state.get("plan_revisions", []))
        edits = list(state.get("plan_edits", []))
        revision = state.get("plan_revision", 0)
        if operations and edit_result["ok"]:
            revision = edit_result["revision"]
            revisions.append({"revision": revision, "stage": "fingering_agent", "plan": plan})
            edits.append({**edit_envelope, "new_revision": revision})
        handoffs = copy.deepcopy(state.get("phrase_handoffs", {}))
        score = score_from_dict(state["score"])
        for phrase_index, phrase in enumerate(score.phrases):
            handoff = handoffs.get(phrase.id)
            if handoff is None or phrase_index == 0:
                continue
            previous = score.phrases[phrase_index - 1]
            previous_ids = set(previous.event_ids)
            previous_actions = [
                action for action in plan.get("actions", [])
                if any(event_id in previous_ids for event_id in action.get("source_event_ids", []))
            ]
            complete = all(
                (not action.get("attack") or action.get("right_finger"))
                and (action.get("mode") != "stopped" or action.get("left_finger"))
                for action in previous_actions
            )
            previous_phrase = handoff.get("previous_phrase", {})
            previous_phrase.update({
                "phrase_id": previous.id,
                "status": "confirmed_readonly" if complete else "route_confirmed_fingering_pending",
                "actions": previous_actions,
            })
            handoff["previous_phrase"] = previous_phrase
            handoff["context_source"] = render_context_source(
                handoff["current_phrase"], previous_phrase=previous_phrase,
                older_context_refs=handoff.get("older_context_refs"),
            )
        return {
            "performance_plan": plan,
            "plan_revision": revision, "plan_revisions": revisions, "plan_edits": edits,
            "fingering_optimization": optimization,
            "phrase_handoffs": handoffs,
            "status": "fingering_optimized",
        }

    def _compile(self, state: GraphState) -> dict:
        score = score_from_dict(state["score"])
        plan = plan_from_dict(state["performance_plan"])
        compiled = self.registry.call("jianzipu.compile", score, plan)
        return {"compiled": compiled, "status": "compiled"}

    def _style_review(self, state: GraphState) -> dict:
        result = self.agent_factory.create("style_critic").run(
            state["job_id"], {
                "canonical_score": state["score"],
                "performance_plan": state["performance_plan"],
                "jianzipu_ir": state["compiled"],
            },
            skill_versions=self.registry.versions(),
        )
        return {"style_review": result.to_dict()}

    def _audit(self, state: GraphState) -> dict:
        score = score_from_dict(state["score"])
        plan = plan_from_dict(state["performance_plan"])
        audit = self.registry.call(
            "jianzipu.audit", score, plan, state["compiled"],
            tolerance_cents=float(state["config"]["audit_tolerance_cents"]),
        )
        return {
            "audit": audit,
            "status": "completed" if audit["status"] == "passed" else "audit_failed",
        }

    @staticmethod
    def _after_audit(state: GraphState) -> str:
        if state["audit"]["status"] == "passed":
            return "done"
        if state.get("repair_count", 0) < int(state["config"]["max_repairs"]):
            return "repair"
        return "done"

    @staticmethod
    def _repair(state: GraphState) -> dict:
        # First deterministic repair expands the playable-position tolerance.
        # Later versions can route individual diagnostics to model-backed agents.
        config = dict(state["config"])
        config["candidate_tolerance_cents"] = min(
            float(config["candidate_tolerance_cents"]) + 15.0, 75.0
        )
        return {
            "config": config,
            "repair_count": state.get("repair_count", 0) + 1,
            "status": "repairing",
        }

    def run(self, abc_text: str, *, config: dict[str, Any] | None = None,
            job_id: str | None = None) -> dict[str, Any]:
        resolved_config = {**DEFAULT_CONFIG, **(config or {})}
        resolved_job_id = job_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        result = self.graph.invoke(
            {
                "job_id": resolved_job_id,
                "abc_text": abc_text,
                "config": resolved_config,
                "phrase_routes": {},
                "repair_count": 0,
                "status": "started",
            },
            config={"configurable": {"thread_id": resolved_job_id}},
        )
        self._write_artifacts(result)
        return result

    def get_state(self, job_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": job_id}})
        return dict(snapshot.values) if snapshot.values else {}

    def _write_artifacts(self, state: GraphState) -> None:
        store = ArtifactStore(self.artifact_root / state["job_id"])
        store.write_text("input.abc", state["abc_text"])
        for name, field in (
            ("config.json", "config"),
            ("canonical_score.json", "score"),
            ("analysis.json", "analysis"),
            ("position_candidates.json", "candidates"),
            ("phrase_handoffs.json", "phrase_handoffs"),
            ("phrase_routes.json", "phrase_routes"),
            ("route_merge_report.json", "route_merge"),
            ("performance_plan.json", "performance_plan"),
            ("fingering_optimization.json", "fingering_optimization"),
            ("jianzipu_ir.json", "compiled"),
            ("style_review.json", "style_review"),
            ("audit_report.json", "audit"),
        ):
            if field in state:
                store.write_json(name, state[field])
        score = state.get("score", {})
        diagnostics = state.get("audit", {}).get("diagnostics", [])
        for revision_item in state.get("plan_revisions", []):
            revision = int(revision_item["revision"])
            store.write_json(f"revisions/r{revision:04d}.json", revision_item["plan"])
            store.write_text(
                f"source/r{revision:04d}.gqs",
                render_plan_source(score, revision_item["plan"], revision=revision,
                                   diagnostics=diagnostics if revision == state.get("plan_revision") else []),
            )
        for edit in state.get("plan_edits", []):
            store.write_json(
                f"patches/r{int(edit['base_revision']):04d}-r{int(edit['new_revision']):04d}.json",
                edit,
            )
        if state.get("performance_plan"):
            store.write_text(
                "source/current.gqs",
                render_plan_source(score, state["performance_plan"],
                                   revision=state.get("plan_revision", 0), diagnostics=diagnostics),
            )
            store.write_text("diagnostics/final.txt",
                             render_diagnostics("current.gqs", score, diagnostics))
        compiled = state.get("compiled", {})
        store.write_json("output.json", {
            "schema_version": compiled.get("schema_version", "1.0"),
            "job_id": state["job_id"],
            "title": state.get("score", {}).get("title"),
            "jianzipu": compiled.get("jianzipu", ""),
            "glyph_events": compiled.get("glyph_events", []),
            "audit": state.get("audit", {}),
        })
        store.write_text("output.md", self._markdown_output(state, compiled.get("jianzipu", "")))
        store.write_json("provenance.json", {
            "job_id": state["job_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": state.get("status"),
            "repair_count": state.get("repair_count", 0),
            "skill_versions": self.registry.versions(),
            "python": sys.version,
            "platform": platform.platform(),
        })

    @staticmethod
    def _markdown_output(state: GraphState, jianzipu: str) -> str:
        score = state.get("score", {})
        audit = state.get("audit", {})
        summary = audit.get("summary", {})
        return (
            f"# {score.get('title', 'Untitled')}\n\n"
            f"| field | value |\n|---|---|\n"
            f"| key | {score.get('key', '')} |\n"
            f"| tuning | {score.get('tuning_name', '')} {score.get('open_midi', '')} |\n"
            f"| audit | {audit.get('status', 'not_run')} |\n"
            f"| pitch match | {summary.get('matched_events', 0)}/{summary.get('compared_events', 0)} |\n\n"
            f"## 减字谱\n\n{jianzipu}\n"
        )
