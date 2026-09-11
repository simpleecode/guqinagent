#!/usr/bin/env python3
"""Audit mapped score files for sparse or truncated jianzi annotations."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ABC_J/agent_training/inferred_v3/inferred_trajectories_train.jsonl"
OUT = ROOT / "ABC_J/agent_training/source_quality_audit_v1"
EMPTY = {"", "—", "-", "null", "None", "[空]", "[减字待填写]"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mapped_path(key: str) -> Path:
    final = ROOT / "ABC_J/final" / key / "jianpu_jianzi_mapped.md"
    if final.exists():
        return final
    return ROOT / "ABC_J/round2/final" / key / "jianpu_jianzi_mapped.md"


def split_md_row(line: str) -> list[str]:
    """Split a markdown row without treating escaped ``\|`` as a separator."""
    parts: list[str] = []
    buffer: list[str] = []
    escaped = False
    for char in line:
        if char == "|" and not escaped:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    parts.append("".join(buffer).strip())
    return parts


def audit_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    title_match = re.search(r"^# 《?(.*?)》? 简谱", text, re.M)
    title = title_match.group(1).strip() if title_match else ""
    url_match = re.search(r"^\|\s*url\s*\|\s*(https?://app\.sitongli\.net/[^|]+?)\s*\|", text, re.M)
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cols = split_md_row(line.strip())
        if cols and cols[0] == "":
            cols = cols[1:]
        if cols and cols[-1] == "":
            cols = cols[:-1]
        if len(cols) < 6 or cols[1] in {"|", r"\|"}:
            continue
        rows.append({"index": int(cols[0]), "jianpu": cols[1], "jianzi": cols[-1], "empty": cols[-1] in EMPTY})
    nonempty = sum(not row["empty"] for row in rows)
    trailing = 0
    for row in reversed(rows):
        if not row["empty"]:
            break
        trailing += 1
    last_nonempty = rows[-trailing - 1]["index"] if trailing < len(rows) else None
    first_trailing = rows[-trailing]["index"] if trailing else None
    ratio = nonempty / len(rows) if rows else 0.0
    if ratio < 0.5:
        classification = "drop_score"
    elif trailing >= 10:
        classification = "trim_trailing"
    else:
        classification = "keep"
    return {
        "score_key": path.parent.name,
        "title": title,
        "score_id": int((re.search(r"score_id\s*\|\s*(\d+)", text) or [None, "0"])[1]),
        "url": url_match.group(1).strip() if url_match else "",
        "path": str(path),
        "notes": len(rows),
        "nonempty": nonempty,
        "nonempty_ratio": ratio,
        "trailing_empty_notes": trailing,
        "last_nonempty_index": last_nonempty,
        "first_trailing_empty_index": first_trailing,
        "classification": classification,
    }


def main() -> int:
    keys = sorted({row["score_key"] for row in read_jsonl(SOURCE)})
    records = [audit_file(mapped_path(key)) for key in keys]
    summary = {
        "training_score_keys": len(records),
        "drop_score_count": sum(r["classification"] == "drop_score" for r in records),
        "trim_trailing_count": sum(r["classification"] == "trim_trailing" for r in records),
        "keep_count": sum(r["classification"] == "keep" for r in records),
        "drop_score_keys": [r["score_key"] for r in records if r["classification"] == "drop_score"],
        "trim_trailing_keys": [r["score_key"] for r in records if r["classification"] == "trim_trailing"],
        "empty_definition": sorted(EMPTY),
        "trailing_threshold_notes": 10,
        "ratio_threshold": 0.5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quality_report.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 映射谱减字质量审计（训练源 176 曲谱）",
        "",
        f"- 整谱剔除（非空比例 < 50%）：{summary['drop_score_count']} 个",
        f"- 截断尾部（非空比例 ≥ 50%，连续空尾 ≥ 10 音）：{summary['trim_trailing_count']} 个",
        f"- 保留：{summary['keep_count']} 个",
        "",
        "## 建议整谱剔除",
        "",
        "| 曲名 | score key | 非空/音符 | 比例 | 连续空尾 | 源文件 | 链接 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in records:
        if r["classification"] == "drop_score":
            lines.append(f"| {r['title']} | `{r['score_key']}` | {r['nonempty']}/{r['notes']} | {r['nonempty_ratio']:.1%} | {r['trailing_empty_notes']} | `{r['path']}` | {r['url']} |")
    lines += ["", "## 建议截断连续空尾", "", "| 曲名 | score key | 非空/音符 | 比例 | 最后非空序号 | 从此序号起空尾 | 空尾音数 | 源文件 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        if r["classification"] == "trim_trailing":
            lines.append(f"| {r['title']} | `{r['score_key']}` | {r['nonempty']}/{r['notes']} | {r['nonempty_ratio']:.1%} | {r['last_nonempty_index']} | {r['first_trailing_empty_index']} | {r['trailing_empty_notes']} | `{r['path']}` |")
    lines += ["", "## 全量明细", "", "完整 176 条记录见同目录 `quality_report.json`。空值定义为 `''`、`—`、`-`、`null`、`[空]`、`[减字待填写]`；小节线不计入音符分母。"]
    (OUT / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
