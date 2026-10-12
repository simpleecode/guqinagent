"""Find stale walk-position warnings in evaluation and preserve score prefixes."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import audit_jianpu_jianzi_pitch as pitch
from run_vllm_eval_queue import balanced_shards

CONTROL = re.compile(r"(?:复|(?:急|缓|少)?(?:进|退|上|下))$")
WARNINGS = ("走手终点与起点相同", "走手方向错误", "走手跨度过大")


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-selection-report", type=Path)
    parser.add_argument("--force-sample", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    inputs = read_rows(args.input)
    predictions = {r["sample_id"]: r for r in read_rows(args.predictions)}
    scores = defaultdict(list)
    hits = defaultdict(list)
    history_by_sample = {}
    for row in inputs:
        scores[row["score_key"]].append(row)
    for rows in scores.values():
        prefix = []
        for row in sorted(rows, key=lambda r: r["runtime_item"]["input"]["event_range"]["start"]):
            history_by_sample[row["sample_id"]] = list(prefix)
            plan = predictions[row["sample_id"]].get("guqinizer", {}).get("plan", {})
            texts = {a["source_index"]: a.get("jianzi_text") or "" for a in plan.get("actions", [])}
            prefix.extend({**n, "jianzi": texts.get(n["index"], "")}
                          for n in row["runtime_item"]["input"]["notes_without_jianzi"])
    for row in inputs:
        pred = predictions[row["sample_id"]]
        notes = row["runtime_item"]["input"]["notes_without_jianzi"]
        event_notes = {n["event_index"]: n for n in notes if n.get("event_index") is not None}
        source_events = {n["index"]: n["event_index"] for n in event_notes.values()}

        def plan_text(plan):
            return {source_events[a["source_index"]]: a.get("jianzi_text") or ""
                    for a in (plan or {}).get("actions", []) if a["source_index"] in source_events}

        reasons = []
        for stage, initial in (("base", {}), ("guqinizer", plan_text((pred.get("base") or {}).get("plan")))):
            texts = dict(initial)
            for turn in (pred.get(stage) or {}).get("trace", []):
                for call in turn.get("tool_results", []):
                    if call.get("name") != "edit_plan" or not call.get("result", {}).get("ok"):
                        continue
                    result = call["result"].get("result", {})
                    # Only applicable previews update the cumulative text.
                    if result.get("valid"):
                        texts.update({int(i): t for i, t in call.get("arguments", {}).get("jianzi_rows", [])
                                      if str(i).isdigit()})
                    for line in str(result.get("text", "")).splitlines():
                        if not any(w in line for w in WARNINGS):
                            continue
                        match = re.match(r"(\d+)｜", line)
                        if not match:
                            continue
                        event = int(match[1])
                        for previous in sorted((i for i in texts if i < event), reverse=True):
                            text = str(texts[previous] or "").strip()
                            if CONTROL.fullmatch(text):
                                replay = {"metadata": row["runtime_item"]["input"]["metadata"],
                                          "open_midi": row["runtime_item"]["input"]["normalized_tuning"]["open_midi"],
                                          "notes": history_by_sample[row["sample_id"]] +
                                          [{**n, "jianzi": texts.get(n.get("event_index"), "")} for n in notes]}
                                fixed = pitch.walk_motion_warnings(replay) + pitch.walk_span_warnings(replay)
                                code = ("walk_endpoint_same_as_start" if WARNINGS[0] in line
                                        else "walk_direction_reversed" if WARNINGS[1] in line
                                        else "walk_span_too_large")
                                if any(w["index"] == event_notes[event]["index"] and w["code"] == code
                                       for w in fixed):
                                    break  # This warning is genuine even after the repair.
                                reasons.append({"kind": "observed_warning_after_control", "stage": stage,
                                                "round": turn.get("round"), "event": event,
                                                "previous_event": previous, "previous_text": text,
                                                "warning": line})
                                break
                            if (pitch.parse_string_numbers(text) or pitch.parse_hui(text) is not None
                                    or pitch.first_token(text, pitch.RIGHT_HAND_TECHNIQUES)
                                    or "泛起" in text or "泛止" in text):
                                break
        # Old evaluator overwrote earlier Guqinizer passes. A final plan or
        # surviving explanation with 复/裸进退 is enough to conservatively
        # include such a phrase; absence of a lost warning is NOT evidence.
        guqinizer = pred.get("guqinizer") or {}
        if pred.get("guqinizer_passes", 1) > 1 and not guqinizer.get("pass_history"):
            control_events = [i for i, t in plan_text(guqinizer.get("plan")).items()
                              if CONTROL.fullmatch(str(t).strip())]
            control_events += [i for i, t in plan_text((pred.get("base") or {}).get("plan")).items()
                               if CONTROL.fullmatch(str(t).strip())]
            if control_events:
                reasons.append({"kind": "lost_earlier_pass_trace_with_control", "events": sorted(set(control_events))})
        if row["sample_id"] in args.force_sample:
            reasons.append({"kind": "explicitly_required_by_user"})
        if reasons:
            hits[row["score_key"]].append({"sample_id": row["sample_id"], "reasons": reasons})
    if args.include_selection_report:
        for score in json.loads(args.include_selection_report.read_text())["scores"]:
            hits[score["score_key"]].append({"sample_id": score["first_sample_id"],
                                           "reasons": [{"kind": "pending_repeat_anchor_suffix"}]})
    affected = []
    prefix_ids = set()
    for key, entries in hits.items():
        rows = sorted(scores[key], key=lambda r: r["runtime_item"]["input"]["event_range"]["start"])
        positions = {r["sample_id"]: i for i, r in enumerate(rows)}
        first = min(positions[h["sample_id"]] for h in entries)
        prefix_ids.update(r["sample_id"] for r in rows[:first])
        affected.append({"score_key": key, "title": rows[0]["runtime_item"]["input"]["metadata"].get("score_title"),
                         "first_sample_id": rows[first]["sample_id"], "first_position": first,
                         "rerun_phrases": len(rows)-first, "preserved_prefix_phrases": first,
                         "affected_phrases": entries})
    for required in args.force_sample:
        if required not in predictions:
            raise ValueError(f"required phrase absent: {required}")
    selected = [r for r in inputs if r["score_key"] in hits]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "input.jsonl"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite prepared rerun: {target}")
    target.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected), encoding="utf-8")
    workers = args.output_dir / "workers"
    workers.mkdir()
    for i, shard in enumerate(balanced_shards(selected, args.workers)):
        if shard:
            (workers/f"predictions_{i:02d}.jsonl").write_text(
                "".join(json.dumps(predictions[r["sample_id"]], ensure_ascii=False)+"\n"
                        for r in shard if r["sample_id"] in prefix_ids), encoding="utf-8")
    report = {"source_predictions": str(args.predictions), "affected_scores": len(affected),
              "directly_flagged_phrase_count": len({h["sample_id"] for es in hits.values() for h in es}),
              "rerun_phrases": sum(s["rerun_phrases"] for s in affected),
              "preserved_prefix_phrases": len(prefix_ids), "required_samples": args.force_sample,
              "scores": affected}
    (args.output_dir/"selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
