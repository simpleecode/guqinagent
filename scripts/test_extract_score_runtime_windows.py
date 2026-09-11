import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("extract_score_runtime_windows.py")
SPEC = importlib.util.spec_from_file_location(
    "extract_score_runtime_windows", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceEventTests(unittest.TestCase):
    def test_decode_errors_are_audited_instead_of_silently_dropped(self):
        score_event = {"off_8": {"off_8": {}}}
        source = [score_event, {"decode_error": "missing argument"}]
        record = {
            "message": {
                "payload": {
                    "event": "enter_decoded",
                    "name": "source_jab_event",
                    "regs": {"x0": {"value": source}},
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            events, reconstruction = MODULE.extract_source_events(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["_source_index"], 0)
        self.assertEqual(reconstruction["runtime_decode_error_indexes"], [1])
        self.assertEqual(
            reconstruction["omitted_source_entries"][0]["value"],
            {"decode_error": "missing argument"},
        )

    def test_unknown_tuning_is_not_fabricated_as_zhengdiao(self):
        args = argparse.Namespace(
            notes_length=0,
            assume_complete_runtime_window=False,
            completeness_method="test",
            input=Path("capture.jsonl"),
            api_response=None,
            score_id=1,
            score_key="test",
            from_id=0,
            from_key="",
            title="test",
            tonic="F",
            tuning_name=None,
            tuning_values=None,
        )
        raw, _ = MODULE.build_outputs([], args)
        self.assertEqual(
            raw["metadata"]["tuning"],
            {
                "name": None,
                "value": None,
                "source": "unknown_not_captured",
            },
        )

    def test_runtime_tuning_is_extracted_from_nested_nab(self):
        record = {
            "message": {
                "payload": {
                    "event": "enter_decoded",
                    "name": "note_tuning_uJk_63e180",
                    "regs": {
                        "x5": {
                            "value": {
                                "off_8!TwoByteString@1234": "正调",
                                "off_c!GrowableList@5678": [0] * 8,
                            }
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            tuning, candidates = MODULE.extract_runtime_tuning(path)
        self.assertEqual(candidates, [{"name": "正调", "value": [0] * 8}])
        self.assertEqual(tuning["source"], "runtime_note_tuning")

    def test_shallow_jab_recovery(self):
        item = {
            "decode_error": "missing argument",
            "shallow_fields": {
                "off_8": {"nested_object": {
                    "off_c_native": "0",
                    "fields": {
                        "off_8": {"value": {"off_1c": ""}},
                        "off_14": {"value": {"off_8": "y:1'/2'"}},
                        "off_18": {"value": None},
                        "off_1c": {"decode_error": "missing argument"},
                    },
                }},
                "off_c": {"value": [{"off_8": {}}]},
                "off_10": {"value": ""},
                "off_14": {"value": ""},
                "off_18": {"value": ""},
                "off_1c": {"value": None},
            },
        }
        event = MODULE.recover_shallow_source_event(item)
        self.assertEqual(event["off_8"]["off_14"]["off_8"], "y:1'/2'")
        self.assertEqual(len(event["off_c"]), 1)


if __name__ == "__main__":
    unittest.main()
