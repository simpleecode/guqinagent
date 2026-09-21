from __future__ import annotations

import json
import unittest
from pathlib import Path

from agents.abc_to_jianzipu.inverse_baseline import build_phrase_baseline
from agents.abc_to_jianzipu.inverse_diff import infer_minimal_patches
from agents.abc_to_jianzipu.reference_parser import parse_reference_actions, AUDIT
from agents.abc_to_jianzipu.trajectory_replay import replay_patches
from agents.abc_to_jianzipu.context_handoff import attach_inferred_handoffs, boundary_state


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "ABC_J" / "final" / "SJAqz3h5" / "jianpu_jianzi_readable.json"


class InverseTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(SAMPLE.read_text(encoding="utf-8"))
        cls.audit = AUDIT.audit(cls.data, 50.0)

    def test_reference_parser_classifies_and_expands(self) -> None:
        actions = parse_reference_actions(self.data, self.audit)
        self.assertEqual(len(actions), len(self.data["notes"]))
        self.assertTrue(any(action.confidence_class == "verified" for action in actions))
        self.assertTrue(any(action.inherited_fields for action in actions))

    def test_attached_harmonic_stop_does_not_leak_to_following_notes(self) -> None:
        data = {
            "metadata": {
                "tonic": "1=F",
                "tuning": {"open_strings": [
                    {"string": 1, "pitch": "C", "octave": 3},
                    {"string": 2, "pitch": "D", "octave": 3},
                    {"string": 3, "pitch": "F", "octave": 3},
                    {"string": 4, "pitch": "G", "octave": 3},
                    {"string": 5, "pitch": "A", "octave": 3},
                    {"string": 6, "pitch": "C", "octave": 4},
                    {"string": 7, "pitch": "D", "octave": 4},
                ]},
            },
            "notes": [
                {"index": 0, "jianpu": "2", "jianzi": "泛起大指七徽勾4弦"},
                {"index": 1, "jianpu": "1", "jianzi": "3弦\n泛止"},
                {"index": 2, "jianpu": "5̣", "jianzi": "名指十徽勾4弦"},
            ],
        }
        actions = parse_reference_actions(data)
        self.assertEqual(actions[1].mode, "harmonic")
        self.assertEqual(actions[2].mode, "stopped")
        context = AUDIT.new_context()
        opens = AUDIT.parse_open_midi(data["metadata"])
        AUDIT.parse_jianzi(data["notes"][0]["jianzi"], opens, context)
        AUDIT.parse_jianzi(data["notes"][1]["jianzi"], opens, context)
        self.assertFalse(context["harmonic"])

    def test_parser_preserves_inherited_outside_hui_label(self) -> None:
        data = {
            "metadata": self.data["metadata"],
            "notes": [
                {"index": 0, "jianpu": "1", "jianzi": "名指徽外勾四弦"},
                {"index": 1, "jianpu": "1", "jianzi": "进"},
            ],
        }
        actions = parse_reference_actions(data)
        self.assertEqual(actions[1].hui, "徽外")
        self.assertIn("hui", actions[1].inherited_fields)

    def test_compound_gestures_are_atomic_context_dependent_attacks(self) -> None:
        data = {
            "metadata": self.data["metadata"],
            "notes": [
                {"index": 0, "jianpu": "1", "jianzi": "掐撮三声"},
                {"index": 1, "jianpu": "2", "jianzi": "掐拨剌二声"},
                {"index": 2, "jianpu": "3", "jianzi": "掐拂歷三声"},
                {"index": 3, "jianpu": "4", "jianzi": "历拂"},
            ],
        }
        actions = parse_reference_actions(data)
        self.assertEqual(
            [action.compound_gesture for action in actions],
            ["掐撮三声", "掐拨剌二声", "掐拂歷三声", "历拂"],
        )
        for action in actions:
            self.assertTrue(action.attack)
            self.assertIsNone(action.right_finger)
            self.assertEqual(action.techniques, [])
            self.assertEqual(action.confidence_class, "verified")
            self.assertIn("compound_gesture", action.explicit_fields)
            pitches, reason = AUDIT.parse_jianzi(
                action.text, AUDIT.parse_open_midi(data["metadata"])
            )
            self.assertEqual(pitches, [])
            self.assertEqual(reason, "context_dependent_compound_gesture")

    def test_compound_gesture_diff_replays_without_substring_technique(self) -> None:
        reference = parse_reference_actions({
            "metadata": self.data["metadata"],
            "notes": [{"index": 7, "jianpu": "1", "jianzi": "掐拨剌三声"}],
        })[0]
        baseline = {"actions": [{
            "source_index": 7, "mode": "stopped", "string": 3, "hui": 7.0,
            "left_finger": "大指", "right_finger": "勾", "attack": True,
            "compound_gesture": None, "techniques": [],
        }]}
        inferred = infer_minimal_patches(baseline, [reference])
        self.assertEqual(
            [patch["patch_type"] for patch in inferred["patches"]],
            ["SET_COMPOUND_GESTURE"],
        )
        replay = replay_patches(baseline, inferred["patches"])
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(replay.actions[0]["compound_gesture"], "掐拨剌三声")
        self.assertEqual(replay.actions[0]["techniques"], [])

    def test_baseline_does_not_contain_reference_text(self) -> None:
        baseline = build_phrase_baseline(self.data, 0, 40)
        serialized = json.dumps(baseline, ensure_ascii=False)
        self.assertNotIn('"jianzi":', serialized)
        self.assertTrue(baseline["actions"])

    def test_minimal_diff_produces_typed_patches(self) -> None:
        baseline = build_phrase_baseline(self.data, 0, 40)
        actions = [
            action for action in parse_reference_actions(self.data, self.audit)
            if 0 <= action.source_index < 40
        ]
        result = infer_minimal_patches(baseline, actions)
        kinds = {patch["patch_type"] for patch in result["patches"]}
        self.assertTrue(kinds <= {
            "REPOSITION", "CHANGE_FINGER", "CHANGE_MODE", "ADD_TECHNIQUE",
            "CHANGE_ATTACK", "SET_COMPOUND_GESTURE", "NO_OP"
        })
        self.assertTrue(result["patches"])

    def test_inferred_patches_replay_deterministically(self) -> None:
        baseline = build_phrase_baseline(self.data, 0, 40)
        actions = [
            action for action in parse_reference_actions(self.data, self.audit)
            if 0 <= action.source_index < 40
        ]
        inferred = infer_minimal_patches(baseline, actions)
        replay = replay_patches(baseline, inferred["patches"])
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(len(replay.applied_patch_ids), len(inferred["patches"]))

    def test_chuo_continuation_and_pre_attack_forms_are_structurally_distinct(self) -> None:
        data = {
            "metadata": self.data["metadata"],
            "notes": [
                {"index": 10, "jianpu": "1", "jianzi": "绰上五徽"},
                {"index": 11, "jianpu": "1", "jianzi": "绰大指五徽六分挑6弦"},
            ],
        }
        actions = parse_reference_actions(data)
        continuation, pre_attack = actions
        self.assertFalse(continuation.attack)
        self.assertIsNone(continuation.right_finger)
        self.assertEqual(continuation.techniques, ["绰"])
        self.assertEqual(continuation.pre_attack_techniques, [])
        self.assertTrue(pre_attack.attack)
        self.assertEqual(pre_attack.right_finger, "挑")
        self.assertEqual(pre_attack.pre_attack_techniques, ["绰"])
        self.assertIn("attack", continuation.explicit_fields)
        self.assertIn("attack", pre_attack.explicit_fields)

    def test_taoqi_is_a_positioned_continuation_technique(self) -> None:
        reference = parse_reference_actions({
            "metadata": self.data["metadata"],
            "notes": [{"index": 12, "jianpu": "1", "jianzi": "名指十徽滔起"}],
        })[0]
        baseline = {"actions": [{
            "source_index": 12, "mode": "stopped", "string": 1, "hui": 9.0,
            "left_finger": "大指", "right_finger": "挑", "attack": True,
            "techniques": [], "pre_attack_techniques": [],
        }]}

        inferred = infer_minimal_patches(baseline, [reference])
        replay = replay_patches(baseline, inferred["patches"])

        self.assertFalse(reference.attack)
        self.assertEqual(reference.techniques, ["滔起"])
        self.assertIn("CHANGE_ATTACK", [p["patch_type"] for p in inferred["patches"]])
        self.assertIn("ADD_TECHNIQUE", [p["patch_type"] for p in inferred["patches"]])
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(replay.actions[0]["techniques"], ["滔起"])

    def test_chuo_patches_replay_attack_and_placement(self) -> None:
        references = parse_reference_actions({
            "metadata": self.data["metadata"],
            "notes": [
                {"index": 10, "jianpu": "1", "jianzi": "绰上五徽"},
                {"index": 11, "jianpu": "1", "jianzi": "绰大指五徽六分挑6弦"},
            ],
        })
        baseline = {"actions": [
            {"source_index": 10, "mode": "stopped", "string": 6, "hui": 6.0,
             "left_finger": "大指", "right_finger": "挑", "attack": True,
             "techniques": [], "pre_attack_techniques": []},
            {"source_index": 11, "mode": "stopped", "string": 6, "hui": 5.6,
             "left_finger": "大指", "right_finger": "挑", "attack": True,
             "techniques": [], "pre_attack_techniques": []},
        ]}
        inferred = infer_minimal_patches(baseline, references)
        types_by_index = {
            index: [patch["patch_type"] for patch in inferred["patches"]
                    if patch["source_index"] == index]
            for index in (10, 11)
        }
        self.assertIn("CHANGE_ATTACK", types_by_index[10])
        pre_patch = next(
            patch for patch in inferred["patches"]
            if patch["source_index"] == 11 and patch["patch_type"] == "ADD_TECHNIQUE"
        )
        self.assertEqual(pre_patch["after"]["placement"], "pre_attack")
        replay = replay_patches(baseline, inferred["patches"])
        self.assertTrue(replay.valid, replay.errors)
        by_index = {action["source_index"]: action for action in replay.actions}
        self.assertFalse(by_index[10]["attack"])
        self.assertIsNone(by_index[10]["right_finger"])
        self.assertEqual(by_index[11]["pre_attack_techniques"], ["绰"])

    def test_chuo_forms_materialize_when_route_has_no_baseline_action(self) -> None:
        references = parse_reference_actions({
            "metadata": self.data["metadata"],
            "notes": [
                {"index": 10, "jianpu": "1", "jianzi": "绰上五徽"},
                {"index": 11, "jianpu": "1", "jianzi": "绰大指五徽六分挑6弦"},
            ],
        })
        inferred = infer_minimal_patches({"actions": []}, references)
        self.assertEqual(inferred["unresolved_differences"], [])
        replay = replay_patches({"actions": []}, inferred["patches"])
        self.assertTrue(replay.valid, replay.errors)
        by_index = {action["source_index"]: action for action in replay.actions}
        self.assertFalse(by_index[10]["attack"])
        self.assertEqual(by_index[10]["hui"], 5.0)
        self.assertTrue(by_index[11]["attack"])
        self.assertEqual(by_index[11]["right_finger"], "挑")
        self.assertEqual(by_index[11]["pre_attack_techniques"], ["绰"])

    def test_phrase_handoff_preserves_full_previous_sequence(self) -> None:
        def item(phrase_id, start, end, action):
            return {
                "phrase_id": phrase_id,
                "input": {"event_range": {"start": start, "end_exclusive": end},
                          "notes_without_jianzi": [
                              {"index": index, "abc": "C", "jianpu": "1", "section": {"number": 1}}
                              for index in range(start, end)]},
                "reference_plan": {"actions": [action]},
            }
        first_action = {"source_index": 1, "mode": "stopped", "string": 4,
                        "hui": 7.6, "left_finger": "大指", "right_finger": "勾",
                        "attack": True, "techniques": ["吟"]}
        second_action = {"source_index": 3, "mode": "open", "string": 5,
                         "hui": None, "left_finger": None, "right_finger": "挑",
                         "attack": True, "techniques": []}
        third_action = {"source_index": 5, "mode": "harmonic", "string": 4,
                        "hui": 7.0, "left_finger": None, "right_finger": "勾",
                        "attack": True, "techniques": []}
        linked = attach_inferred_handoffs([
            item("p0001", 0, 2, first_action), item("p0002", 2, 4, second_action),
            item("p0003", 4, 6, third_action),
        ])
        first_handoff = linked[0]["input"]["phrase_handoff"]
        self.assertNotIn("previous_phrase", first_handoff)
        self.assertNotIn("older_context_refs", first_handoff)
        previous = linked[1]["input"]["phrase_handoff"]["previous_phrase"]
        self.assertEqual(previous["phrase_id"], "p0001")
        self.assertEqual(previous["actions"], [first_action])
        self.assertEqual(len(previous["notes"]), 2)
        self.assertTrue(linked[1]["input"]["phrase_handoff"]["current_phrase"])
        self.assertNotIn("right_lookahead", linked[1]["input"]["phrase_handoff"])
        older_refs = linked[2]["input"]["phrase_handoff"]["older_context_refs"]
        self.assertEqual([item["phrase_id"] for item in older_refs], ["p0001"])
        self.assertTrue(older_refs[0]["context_ref"].endswith("/phrases/p0001"))


if __name__ == "__main__":
    unittest.main()
