"""Build the fixed Complex Fingering Handbook artifact.

Selects the knowledge entries that actually occur in the training-half
reference annotations using the **teacher's own matcher semantics**
(``generate_teacher_tool_trajectories.matched_compound_gesture_knowledge``):
every slash-separated alias is indexed, matching runs against the rendered
reference text (``jianzi_text``/``text``, not the sparse structured
``techniques`` field), longest alias wins, and a longer match subsumes its
substrings (掐撮三声 subsumes 掐撮).  One alias may select several entries
(撮 and 大撮/撮); counts aggregate by the knowledge row's indexed name.

The artifact is checked in and loaded verbatim by ``handbook.py``; it is
deliberately outside GEPA's mutation surface.  Only definitional knowledge is
kept.  No melody-conditional heuristics, no event-level advice, no training
examples.

Rerun to regenerate (e.g. after a data re-split):

    python -m baseline.gepa_guqin.build_handbook
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    COMPLEX_GESTURE_KNOWLEDGE_MATCHES,
)

DEFAULT_SOURCE = (ROOT / "ABC_J/agent_training/renderfix_train_test_20261007"
                  / "pitch_eligible_two_or_half/inferred_trajectories_train.jsonl")
KNOWLEDGE = (ROOT / "agents/abc_to_jianzipu/knowledge"
             / "complex_fingering_explanations_v3_with_effects.jsonl")
HEADER = "【复杂指法手册（固定参考知识，不属于可优化的策略提示词）】"


def annotation_entry_hits(source: Path) -> Counter:
    """Teacher-semantics knowledge-entry hits per phrase, by indexed name."""
    aliases = sorted(COMPLEX_GESTURE_KNOWLEDGE_MATCHES, key=len, reverse=True)
    indexed: dict[str, dict[str, str]] = {}
    for entries in COMPLEX_GESTURE_KNOWLEDGE_MATCHES.values():
        for entry in entries:
            indexed[entry["indexed_name"]] = entry
    counts: Counter = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reference_text = "\n".join(
            str(action.get("jianzi_text") or action.get("text") or "")
            for action in row["reference_plan"]["actions"]
        )
        matched: list[str] = []
        for alias in aliases:
            if alias not in reference_text:
                continue
            if any(alias in prior for prior in matched):
                continue
            matched.append(alias)
        for alias in matched:
            for entry in COMPLEX_GESTURE_KNOWLEDGE_MATCHES[alias]:
                counts[entry["indexed_name"]] += 1
    return counts


def build(source: Path, out_path: Path) -> dict:
    counts = annotation_entry_hits(source)
    knowledge = [json.loads(line) for line in
                 KNOWLEDGE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_name = {entry["name"]: entry for entry in knowledge}
    selected = [by_name[name] for name in sorted(counts, key=lambda n: -counts[n])
                if name in by_name]

    lines = [HEADER, ""]
    category_order = ["特殊/左右手配合指法", "左手指法", "右手指法", "一般性指法/谱字符号"]
    for category in category_order:
        group = [entry for entry in selected if entry["category"] == category]
        if not group:
            continue
        lines.append(f"◆ {category}")
        lines.append("")
        for entry in group:
            lines.append(f"【{entry['name']}】（训练标注 {counts[entry['name']]} 个段落命中）")
            lines.append(entry["explanation"].strip())
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    manifest = {
        "knowledge_source": str(KNOWLEDGE),
        "annotation_source": str(source),
        "matcher": "teacher matched_compound_gesture_knowledge semantics "
                   "(alias table + longest-match over rendered reference text + "
                   "substring subsumption; counts by indexed knowledge name)",
        "selected_entries": [entry["name"] for entry in selected],
        "selected_count": len(selected),
        "phrase_hit_counts": dict(counts.most_common()),
    }
    (out_path.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="training-half trajectories used for the filter")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "baseline/gepa_guqin/knowledge/complex_fingering_handbook.md")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
