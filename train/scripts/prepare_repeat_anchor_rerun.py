"""Select score suffixes exposed to an unresolved 从ㄱ再作 reference."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_vllm_eval_queue import balanced_shards


def anchor(text):
    return any(mark in str(text or "") for mark in ("再作起点", "再作标记", "再做起点", "再做标记"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    inputs = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    predictions = {r["sample_id"]: r for r in (json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip())}
    scores = defaultdict(list)
    for row in inputs:
        scores[row["score_key"]].append(row)
    affected = []
    for score, rows in scores.items():
        rows.sort(key=lambda r: r["runtime_item"]["input"]["event_range"]["start"])
        history_anchor = False
        hits = []
        for position, row in enumerate(rows):
            pred = predictions[row["sample_id"]]
            notes = row["runtime_item"]["input"]["notes_without_jianzi"]
            ordered_events = [n["event_index"] for n in notes if n.get("event_index") is not None]
            source_events = {n["index"]: n["event_index"] for n in notes if n.get("event_index") is not None}

            def orphaned(texts):
                seen = history_anchor
                missing = []
                for event in ordered_events:
                    text = str(texts.get(event) or "")
                    if "从ㄱ再作" in text and not seen:
                        missing.append(event)
                    if anchor(text):
                        seen = True
                return missing

            def plan_text(plan):
                return {source_events[a["source_index"]]: a.get("jianzi_text") or a.get("text") or ""
                        for a in plan.get("actions", []) if a.get("source_index") in source_events}

            phrase_hits = set()
            for stage, initial in (("base", {}), ("guqinizer", plan_text((pred.get("base") or {}).get("plan") or {}))):
                texts = dict(initial)
                for turn in (pred.get(stage) or {}).get("trace", []):
                    for call in turn.get("tool_results", []):
                        if call.get("name") != "edit_plan" or not call.get("result", {}).get("ok"):
                            continue
                        for event, text in call.get("arguments", {}).get("jianzi_rows", []):
                            if str(event).isdigit():
                                texts[int(event)] = text
                        phrase_hits.update(orphaned(texts))
                phrase_hits.update(orphaned(plan_text((pred.get(stage) or {}).get("plan") or {})))
            if phrase_hits:
                hits.append({"sample_id": row["sample_id"], "position": position,
                             "event_indices": sorted(phrase_hits)})
            history_anchor = history_anchor or any(anchor(t) for t in plan_text(pred["guqinizer"]["plan"]).values())
        if hits:
            first = hits[0]["position"]
            title = rows[0]["runtime_item"]["input"]["metadata"].get("score_title", score)
            affected.append({"score_key": score, "title": title, "first_sample_id": hits[0]["sample_id"],
                             "first_position": first, "rerun_phrases": len(rows) - first,
                             "preserved_prefix_phrases": first, "affected_phrases": hits})
    selected_scores = {r["score_key"]: r for r in affected}
    selected = [r for r in inputs if r["score_key"] in selected_scores]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    worker_dir = args.output_dir / "workers"
    worker_dir.mkdir(exist_ok=True)
    prefix_ids = set()
    for entry in affected:
        prefix_ids.update(r["sample_id"] for r in scores[entry["score_key"]][:entry["first_position"]])
    for worker, shard in enumerate(balanced_shards(selected, args.workers)):
        if shard:
            path = worker_dir / f"predictions_{worker:02d}.jsonl"
            if path.exists():
                raise RuntimeError(f"refusing to replace existing rerun predictions: {path}")
            path.write_text("".join(json.dumps(predictions[r["sample_id"]], ensure_ascii=False) + "\n"
                                    for r in shard if r["sample_id"] in prefix_ids))
    report = {"affected_scores": len(affected), "affected_phrase_count": sum(len(r["affected_phrases"]) for r in affected),
              "rerun_phrases": sum(r["rerun_phrases"] for r in affected), "preserved_prefix_phrases": len(prefix_ids),
              "source_predictions": str(args.predictions), "scores": affected}
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
