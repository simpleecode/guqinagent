#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from pathlib import Path


def load_data(path):
    with Path(path).open(encoding="utf-8") as fp:
        return json.load(fp)


def event_has_jianzi(event):
    raw = event.get("jianzi_raw")
    return isinstance(raw, list) and len(raw) > 0


def fuse_events(datasets, labels):
    base = deepcopy(datasets[0])
    fused = []
    max_len = max(len(item.get("events", [])) for item in datasets)
    conflicts = []

    for index in range(max_len):
        candidates = []
        for label, data in zip(labels, datasets):
            events = data.get("events", [])
            if index < len(events):
                ev = events[index]
                candidates.append((label, ev))

        if not candidates:
            continue

        selected_label, selected = candidates[0]
        for label, ev in candidates:
            if event_has_jianzi(ev):
                selected_label, selected = label, ev
                break

        out = deepcopy(selected)
        out["fusion_sources"] = [
            {
                "label": label,
                "has_jianzi": event_has_jianzi(ev),
                "jianpu": ev.get("jianpu"),
                "reconstruction_source": ev.get("reconstruction_source"),
            }
            for label, ev in candidates
        ]
        out["fusion_selected_source"] = selected_label

        jianzi_blobs = {
            json.dumps(ev.get("jianzi_raw"), ensure_ascii=False, sort_keys=True)
            for _, ev in candidates
            if event_has_jianzi(ev)
        }
        if len(jianzi_blobs) > 1:
            conflicts.append(index)

        fused.append(out)

    base["events"] = fused
    stats = base.setdefault("stats", {})
    total = len(fused)
    jianpu = sum(1 for ev in fused if ev.get("jianpu"))
    jianzi = sum(1 for ev in fused if event_has_jianzi(ev))
    aligned = sum(1 for ev in fused if ev.get("jianpu") and event_has_jianzi(ev))
    unaligned = [ev.get("index", idx) for idx, ev in enumerate(fused) if ev.get("jianpu") and not event_has_jianzi(ev)]
    stats.update({
        "total_events": total,
        "jianpu_events": jianpu,
        "jianzi_events": jianzi,
        "successful_alignments": aligned,
        "unaligned_count": len(unaligned),
        "unaligned_event_indexes": unaligned,
        "fusion": {
            "input_count": len(datasets),
            "input_labels": labels,
            "conflict_indexes": conflicts,
            "coverage_before": [
                {
                    "label": label,
                    "total_events": len(data.get("events", [])),
                    "jianzi_events": sum(1 for ev in data.get("events", []) if event_has_jianzi(ev)),
                    "unaligned_count": data.get("stats", {}).get("unaligned_count"),
                    "note_call_only_events": data.get("stats", {}).get("reconstruction", {}).get("note_call_only_events"),
                }
                for label, data in zip(labels, datasets)
            ],
        },
    })
    stats["capture_completeness"] = {
        "method": "multi-run runtime WZa fusion",
        "complete": False,
        "note": "Fusion can fill jianzi only when at least one input capture contains the matching WZa event.",
    }
    base["anomalies"] = [{
        "type": "jianpu_without_jianzi_after_fusion",
        "indexes": unaligned,
        "message": "No input capture contained a WZa jianzi event for these jianpu events.",
    }] if unaligned else []
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Path to data.json. May be repeated.")
    parser.add_argument("--label", action="append", default=[], help="Optional label per input.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    datasets = [load_data(path) for path in args.input]
    labels = args.label or [Path(path).parent.parent.name for path in args.input]
    if len(labels) != len(datasets):
        raise SystemExit("--label count must match --input count")

    fused = fuse_events(datasets, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = fused["stats"]
    print(json.dumps({
        "inputs": stats["fusion"]["coverage_before"],
        "fused": {
            "total_events": stats["total_events"],
            "jianpu_events": stats["jianpu_events"],
            "jianzi_events": stats["jianzi_events"],
            "successful_alignments": stats["successful_alignments"],
            "unaligned_count": stats["unaligned_count"],
            "conflicts": len(stats["fusion"]["conflict_indexes"]),
            "output": str(args.out),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
