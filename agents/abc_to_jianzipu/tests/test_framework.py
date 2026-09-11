from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.abc_to_jianzipu.abc_parser import parse_abc
from agents.abc_to_jianzipu.backends import AnthropicCompatibleBackend
from agents.abc_to_jianzipu.compiler import hui_label
from agents.abc_to_jianzipu.models import TaskResult
from agents.abc_to_jianzipu.orchestrator import AbcToJianzipuOrchestrator
from agents.abc_to_jianzipu.pitch_candidates import generate_candidates
from agents.abc_to_jianzipu.route_planner import merge_phrase_routes_report, top_k_routes
from agents.abc_to_jianzipu.tuning_planner import decide_tuning, rank_tunings


SAMPLE_ABC = """\
X:1
T:框架测试
M:4/4
L:1/8
Q:1/4=60
K:C
% <一 起>
C2 D2 E2 F2 | G4-G4 |
% <二 收>
A2 G2 z4 |
"""


class ParserTests(unittest.TestCase):
    def test_sections_ties_and_rests(self) -> None:
        score = parse_abc(SAMPLE_ABC, phrase_bars=8)
        self.assertEqual(score.title, "框架测试")
        self.assertEqual([section.marker for section in score.sections], ["<一 起>", "<二 收>"])
        self.assertEqual(len(score.events), 8)
        tied = next(event for event in score.events if event.abc == "G4-G4")
        self.assertEqual(tied.duration_ticks, 3840)
        self.assertEqual(score.events[-1].kind, "rest")

    def test_candidates_are_auditable(self) -> None:
        score = parse_abc(SAMPLE_ABC)
        candidates = generate_candidates(score)
        self.assertTrue(candidates)
        for event_id, items in candidates.items():
            self.assertTrue(items, event_id)
            self.assertTrue(all(item.event_id == event_id for item in items))

    def test_tuning_candidates_are_ranked_and_complete(self) -> None:
        score = parse_abc(SAMPLE_ABC)
        ranked = rank_tunings(score)
        self.assertGreaterEqual(len(ranked), 5)
        self.assertEqual([item["rank"] for item in ranked], list(range(1, len(ranked) + 1)))
        self.assertTrue(all(len(item["open_midi"]) == 7 for item in ranked))

    def test_retuning_is_a_thresholded_decision(self) -> None:
        score = parse_abc(SAMPLE_ABC)
        solo = decide_tuning(score, performance_context="solo", open_tone_demand="high")
        ensemble = decide_tuning(score, performance_context="ensemble", open_tone_demand="low")
        self.assertIn("standard_assessment", solo)
        self.assertIn("best_alternative", solo)
        self.assertLess(solo["required_gain"], ensemble["required_gain"])

    def test_repository_leading_tie_spelling(self) -> None:
        score = parse_abc("X:1\nM:1/4\nL:1/16\nK:C\nC4 -C4 | D4\n")
        self.assertEqual(len(score.events), 2)
        self.assertEqual(score.events[0].abc, "C4-C4")
        self.assertEqual(score.events[0].duration_ticks, 1920)


class CompilerTests(unittest.TestCase):
    def test_chinese_hui_label_is_not_corrupted(self) -> None:
        self.assertEqual(hui_label(6.2), "六徽二分")
        self.assertNotIn("�", hui_label(6.2))


class BackendTests(unittest.TestCase):
    def test_model_json_parser_accepts_plain_and_fenced_json(self) -> None:
        expected = {"status": "completed", "output": {"ok": True}}
        self.assertEqual(
            AnthropicCompatibleBackend._parse_json('{"status":"completed","output":{"ok":true}}'),
            expected,
        )
        self.assertEqual(
            AnthropicCompatibleBackend._parse_json(
                '说明\n```json\n{"status":"completed","output":{"ok":true}}\n```'
            ),
            expected,
        )


class OrchestratorTests(unittest.TestCase):
    def test_end_to_end_map_reduce_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            orchestrator = AbcToJianzipuOrchestrator(temporary)
            result = orchestrator.run(SAMPLE_ABC, job_id="test-job")
            self.assertEqual(result["audit"]["status"], "failed")
            self.assertEqual(result["audit"]["summary"]["match_rate"], 1.0)
            self.assertFalse(result["audit"]["checks"]["fingering_complete"])
            self.assertEqual(set(result["phrase_routes"]), {"p0001", "p0002"})
            self.assertEqual(len(result["route_merge"]["selected_routes"]), 2)
            self.assertEqual(
                result["route_merge"]["total_cost"],
                round(
                    result["route_merge"]["local_cost"]
                    + result["route_merge"]["boundary_cost"], 6,
                ),
            )
            self.assertEqual(result["route_merge"]["boundaries"][0]["available_beats"], 4.0)
            self.assertEqual(set(result["phrase_handoffs"]), {"p0001", "p0002"})
            self.assertNotIn("previous_phrase", result["phrase_handoffs"]["p0001"])
            self.assertNotIn("older_context_refs", result["phrase_handoffs"]["p0001"])
            self.assertTrue(result["phrase_handoffs"]["p0002"]["current_phrase"])
            self.assertEqual(
                result["phrase_handoffs"]["p0002"]["previous_phrase"]["phrase_id"],
                "p0001",
            )
            self.assertNotIn("right_lookahead", result["phrase_handoffs"]["p0002"])
            self.assertIn(
                "<前一段：p0001，已确认，只读>",
                result["phrase_handoffs"]["p0002"]["context_source"],
            )
            output = Path(temporary) / "test-job" / "output.md"
            self.assertTrue(output.exists())
            text = output.read_text(encoding="utf-8")
            self.assertIn("<一 起>", text)
            self.assertIn("待定", text)
            self.assertNotIn("�", text)
            self.assertEqual(orchestrator.get_state("test-job")["status"], "audit_failed")
            source = Path(temporary) / "test-job" / "source" / "current.gqs"
            self.assertTrue(source.exists())
            source_text = source.read_text(encoding="utf-8")
            self.assertIn("序号@事件ID｜目标音｜时值", source_text)
            self.assertIn("起音：", source_text)
            self.assertIn("谱面减字", source_text)
            self.assertNotIn("完整减字", source_text)
            self.assertIn("右手待定", source_text)
            self.assertIn("按指待定", source_text)
            self.assertTrue(
                (Path(temporary) / "test-job" / "route_merge_report.json").exists()
            )

    def test_global_merger_rejects_a_missing_phrase_route(self) -> None:
        score = parse_abc(SAMPLE_ABC, phrase_bars=8)
        candidates = generate_candidates(score)
        phrase = score.phrases[0]
        routes = top_k_routes(
            score, phrase.id,
            {event_id: candidates[event_id] for event_id in phrase.event_ids if event_id in candidates},
        )
        candidate_index = {
            candidate.candidate_id: candidate
            for items in candidates.values() for candidate in items
        }
        with self.assertRaisesRegex(ValueError, "missing_phrase_routes"):
            merge_phrase_routes_report(
                score, {phrase.id: routes}, candidate_index,
            )

    def test_model_backend_is_called_for_bounded_roles(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.roles: list[str] = []

            def run(self, task, prompt, tools=None) -> TaskResult:
                self.roles.append(task.agent_role)
                return TaskResult(
                    task_id=task.task_id,
                    agent_role=task.agent_role,
                    status="completed",
                    output={"role": task.agent_role},
                )

        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            result = AbcToJianzipuOrchestrator(
                temporary, backend=backend
            ).run(SAMPLE_ABC, job_id="model-routing-test")
            self.assertEqual(
                backend.roles[:4],
                ["tuning_planner", "score_analyst", "fingering_agent", "style_critic"],
            )
            self.assertLessEqual(backend.roles.count("fingering_agent"), 2)
            self.assertLessEqual(backend.roles.count("style_critic"), 2)
            self.assertEqual(result["analysis"]["status"], "completed")
            self.assertEqual(result["fingering_optimization"]["status"], "completed")
            self.assertEqual(result["style_review"]["status"], "completed")

    def test_optimizer_can_call_candidate_and_pitch_tools(self) -> None:
        class ToolUsingBackend:
            def run(self, task, prompt, tools=None) -> TaskResult:
                output = {}
                if task.agent_role == "fingering_agent":
                    expanded = tools.invoke(task, "expand_context", {
                        "phrase_id": "p0001",
                    })
                    if not expanded["ok"] or not expanded["result"]["readonly"]:
                        raise AssertionError(expanded)
                    queried = tools.invoke(task, "get_pitch_candidates", {
                        "event_id": "n00001", "类型": "按音",
                    })
                    if queried["result"]["format"] != "guqin-candidate-table-1.0":
                        raise AssertionError(queried)
                    if "序号｜方式｜弦徽｜实得MIDI｜可信度" not in queried["result"]["text"]:
                        raise AssertionError(queried)
                    candidate = queried["result"]["top_candidate"]
                    checked = tools.invoke(task, "calculate_guqin_pitch", {
                        "event_id": "n00001",
                        "string": candidate["string"],
                        "hui": candidate["hui"],
                        "mode": candidate["mode"],
                    })
                    if not checked["ok"]:
                        raise AssertionError(checked)
                    preview = tools.invoke(task, "edit_plan", {
                        "base_revision": task.payload.get("plan_revision", 0),
                        "operations": [{
                            "op": "set_fields", "action_id": "a00001",
                            "expected": {},
                            "fields": {"mode": candidate["mode"], "string": candidate["string"],
                                       "hui": candidate["hui"]},
                        }],
                    })
                    if not preview["ok"] or not preview["result"]["commit_required"]:
                        raise AssertionError(preview)
                    replacements = []
                    for action in task.payload["performance_plan"]["actions"]:
                        fields = {
                            "action_id": action["action_id"],
                            "mode": action["mode"],
                            "string": action["string"],
                            "hui": action.get("hui"),
                            "left_finger": "大指" if action["mode"] == "stopped" else None,
                            "right_finger": "挑" if action.get("attack", True) else None,
                            "reason": "test fingering assignment",
                        }
                        if action["action_id"] == "a00001":
                            fields.update({
                                "mode": candidate["mode"], "string": candidate["string"],
                                "hui": candidate["hui"],
                                "left_finger": "大指" if candidate["mode"] == "stopped" else None,
                            })
                        replacements.append(fields)
                    output = {"replacements": replacements}
                return TaskResult(
                    task_id=task.task_id, agent_role=task.agent_role,
                    status="completed", output=output,
                )

        with tempfile.TemporaryDirectory() as temporary:
            result = AbcToJianzipuOrchestrator(
                temporary, backend=ToolUsingBackend()
            ).run(SAMPLE_ABC, job_id="tool-routing-test")
            optimization = result["fingering_optimization"]
            trace_step = next(
                step for step in optimization["trace"]
                if step["type"] == "tool_broker_trace"
            )
            trace = trace_step["content"]
            self.assertEqual(
                [item["tool"] for item in trace],
                ["expand_context", "get_pitch_candidates", "calculate_guqin_pitch", "edit_plan"],
            )
            self.assertEqual(optimization["application"][0]["status"], "applied")
            self.assertEqual(result["audit"]["status"], "passed")

    def test_tuning_agent_can_compare_and_select(self) -> None:
        class TuningBackend:
            def run(self, task, prompt, tools=None) -> TaskResult:
                output = {}
                if task.agent_role == "tuning_planner":
                    result = tools.invoke(task, "assess_retuning", {
                        "performance_context": "solo", "open_tone_demand": "auto"
                    })
                    chosen = result["result"]
                    output = {
                        "retune": chosen["retune"],
                        "selected_name": chosen["selected_name"], "open_midi": chosen["open_midi"],
                        "reason_codes": ["TOOL_RANKED"], "alternatives": [],
                    }
                return TaskResult(
                    task_id=task.task_id, agent_role=task.agent_role,
                    status="completed", output=output,
                )

        with tempfile.TemporaryDirectory() as temporary:
            result = AbcToJianzipuOrchestrator(
                temporary, backend=TuningBackend()
            ).run(SAMPLE_ABC, job_id="tuning-routing-test")
            self.assertEqual(result["tuning_decision"]["reason_codes"], ["TOOL_RANKED"])
            self.assertEqual(len(result["score"]["open_midi"]), 7)


if __name__ == "__main__":
    unittest.main()
