from __future__ import annotations

import json
import re
from typing import Any

SCHEMA = "teacher-gqs-1.2"
LEGACY_SCHEMA = "teacher-gqs-1.1"
FIELDS = ("index", "jianpu", "jianpu_alt", "abc", "duration", "lyric", "jianzi", "jianzi_text", "section")
# First cell is the contiguous agent-facing ordinal; remaining cells preserve
# null/empty values so parsing remains lossless.
COMPACT_FIELDS = ("jianpu", "jianpu_alt", "abc", "duration", "lyric", "jianzi", "jianzi_text")
ROW_FIELDS = ("event_index",) + COMPACT_FIELDS


def _is_bar(note: dict[str, Any]) -> bool:
    return (str(note.get("jianpu") or "").strip() == "|"
            or str(note.get("abc") or "").strip() == "|"
            or note.get("duration") == "小节线")


def _marker(section: dict[str, Any] | None) -> str:
    section = section or {}
    marker = str(section.get("marker") or "").strip()
    if marker:
        return marker
    label = str(section.get("label") or "").strip()
    title = str(section.get("title") or "").strip()
    return f"<{label}{(' ' + title) if title else ''}>" if label else ""


def render_teacher_gqs(data: dict[str, Any]) -> str:
    """Render compact GQS 1.2; bars/sections are structural records."""
    metadata = data.get("metadata") or {}
    notes = list(data.get("notes") or [])
    lines = [f"GQS｜{SCHEMA}", f"谱名｜{metadata.get('score_title', '未命名')}",
             "元数据｜" + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))]
    event_sources: list[int] = []
    bar_sources: list[dict[str, Any]] = []
    section_defs: list[dict[str, Any] | None] = []
    section_ids: dict[str, int] = {}
    row_section_ids: list[int | None] = []
    ordinal = 0
    current_marker = None
    header_written = False
    for note in notes:
        marker = _marker(note.get("section"))
        if marker and marker != current_marker:
            lines.extend(["", marker, "序号｜简谱｜ABC｜时值｜减字"])
            current_marker = marker
            header_written = True
        elif not header_written:
            lines.extend(["", "序号｜简谱｜ABC｜时值｜减字"])
            header_written = True
        if _is_bar(note):
            lines.append("小节线")
            if note.get("index") is not None:
                bar_sources.append({
                    "index": int(note["index"]),
                    "jianpu": note.get("jianpu"),
                    "jianpu_alt": note.get("jianpu_alt"),
                    "abc": note.get("abc"),
                    "duration": note.get("duration"),
                    "lyric": note.get("lyric"),
                    "jianzi": note.get("jianzi"),
                    "jianzi_text": note.get("jianzi_text"),
                })
            section = note.get("section")
            key = json.dumps(section, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in section_ids:
                section_ids[key] = len(section_defs)
                section_defs.append(section)
            row_section_ids.append(section_ids[key])
            continue
        event_sources.append(int(note["index"]))
        values = [ordinal] + [note.get(field) for field in COMPACT_FIELDS]
        lines.append("音｜" + json.dumps(values, ensure_ascii=False, separators=(",", ":")))
        section = note.get("section")
        key = json.dumps(section, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in section_ids:
            section_ids[key] = len(section_defs)
            section_defs.append(section)
        row_section_ids.append(section_ids[key])
        ordinal += 1
    all_sources = [int(note["index"]) for note in notes if note.get("index") is not None]
    contiguous = bool(all_sources) and all(b - a == 1 for a, b in zip(all_sources, all_sources[1:]))
    mapping: dict[str, Any] = {"线": [bar["index"] for bar in bar_sources]}
    if contiguous:
        mapping["音起点"] = all_sources[0]
    else:
        # Preserve losslessness for legacy/sparse sources without imposing a
        # large mapping on the normal contiguous score corpus.
        mapping["音"] = event_sources
    # Run-length encode section ids; section start metadata remains in defs.
    runs: list[list[int]] = []
    for pos, section_id in enumerate(row_section_ids):
        if runs and runs[-1][2] == section_id:
            runs[-1][1] = pos + 1
        else:
            runs.append([pos, pos + 1, section_id])
    mapping["节定义"] = section_defs
    mapping["节段"] = runs
    # A standard bar is reconstructed from its structural row. Keep field
    # payloads only when source data carries a nonstandard value.
    standard_bar = {"jianpu": "|", "jianpu_alt": None, "abc": "|",
                    "duration": "小节线", "lyric": "", "jianzi": "",
                    "jianzi_text": None}
    if any(any(bar.get(key) != value for key, value in standard_bar.items())
           for bar in bar_sources):
        mapping["线字段"] = bar_sources
    lines.insert(3, "索引映射｜" + json.dumps(
        mapping, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _parse_v12(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    metadata = None
    mapping: dict[str, Any] = {"音": [], "线": []}
    rows: list[tuple[str, Any]] = []
    active_section: dict[str, Any] | None = None
    section_count = 0
    for line_number, line in enumerate(lines[1:], 2):
        if line.startswith("元数据｜"):
            metadata = json.loads(line.split("｜", 1)[1])
        elif line.startswith("索引映射｜"):
            mapping = json.loads(line.split("｜", 1)[1])
        elif line == "小节线":
            rows.append(("bar", None))
        elif line.startswith("音｜"):
            values = json.loads(line.split("｜", 1)[1])
            if len(values) != len(ROW_FIELDS):
                raise ValueError(f"wrong compact row at line {line_number}")
            rows.append(("note", values))
        elif line.startswith("序号｜") or not line.strip() or line.startswith("谱名｜"):
            continue
        elif re.fullmatch(r"<[^>]*>", line.strip()):
            marker = line.strip()
            parts = marker[1:-1].split(None, 1)
            section_count += 1
            active_section = {"number": section_count, "label": parts[0] if parts else "",
                              "title": parts[1] if len(parts) > 1 else "",
                              "start": True, "marker": marker}
            rows.append(("section", active_section))
        else:
            raise ValueError(f"unexpected GQS 1.2 line {line_number}: {line[:40]}")
    if metadata is None:
        raise ValueError("GQS has no metadata")
    notes: list[dict[str, Any]] = []
    event_pos = bar_pos = 0
    row_pos = 0
    section_defs = mapping.get("节定义") or []
    section_seq = mapping.get("节序") or []
    section_runs = mapping.get("节段") or []

    def section_id_at(position: int) -> int | None:
        if section_runs:
            for start, end, section_id in section_runs:
                if int(start) <= position < int(end):
                    return int(section_id)
            return None
        return section_seq[position] if position < len(section_seq) else None

    def row_section() -> dict[str, Any] | None:
        nonlocal row_pos
        section_id = section_id_at(row_pos)
        row_pos += 1
        if section_id is None or not isinstance(section_id, int) or section_id >= len(section_defs):
            return None
        section = section_defs[section_id]
        return dict(section) if isinstance(section, dict) else section

    for kind, payload in rows:
        if kind == "section":
            active_section = dict(payload)
            continue
        if kind == "bar":
            bar_info = ((mapping.get("线字段") or mapping.get("线") or [])[bar_pos]
                        if bar_pos < len(mapping.get("线字段") or mapping.get("线") or []) else {})
            if isinstance(bar_info, dict):
                source = bar_info.get("index")
            else:
                source = bar_info
            bar_pos += 1
            if isinstance(bar_info, dict):
                bar_note = {field: bar_info.get(field) for field in FIELDS if field != "section"}
                bar_note["index"] = source
            else:
                bar_note = {"index": source, "jianpu": "|", "jianpu_alt": None,
                            "abc": "|", "duration": "小节线", "lyric": "",
                            "jianzi": "", "jianzi_text": None}
            bar_note["section"] = row_section()
            notes.append(bar_note)
            if source is not None:
                source_cursor = int(source) + 1
            continue
        # payload[0] is the contiguous ordinal; source indexes are recovered
        # only through the internal mapping so the model-facing index stays
        # independent of sparse source rows.
        values = dict(zip(COMPACT_FIELDS, payload[1:]))
        if mapping.get("音"):
            source = mapping["音"][event_pos] if event_pos < len(mapping["音"]) else int(payload[0])
        else:
            # For the normal contiguous corpus, bars are the only skipped
            # source rows.  Advance a source cursor over the rendered row
            # sequence while retaining the explicit bar indexes.
            source = locals().get("source_cursor", mapping.get("音起点", 0))
            source = int(source)
        event_pos += 1
        values = {field: values.get(field) for field in COMPACT_FIELDS}
        values["section"] = row_section()
        notes.append({"index": source, **values})
        source_cursor = source + 1
    return {"metadata": metadata, "notes": notes}


def parse_teacher_gqs(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty teacher GQS")
    if lines[0] == f"GQS｜{SCHEMA}":
        return _parse_v12(text)
    if lines[0] != f"GQS｜{LEGACY_SCHEMA}":
        raise ValueError("unsupported teacher GQS schema")
    metadata = None
    fields = None
    notes = []
    for line_number, line in enumerate(lines[1:], 2):
        if line.startswith("元数据｜"):
            metadata = json.loads(line.split("｜", 1)[1])
        elif line.startswith("列｜"):
            fields = tuple(line.split("｜")[1:])
            if fields != FIELDS:
                raise ValueError(f"unexpected GQS columns at line {line_number}")
        elif line.startswith("音｜"):
            if fields is None:
                raise ValueError(f"GQS note before columns at line {line_number}")
            cells = json.loads(line.split("｜", 1)[1])
            if len(cells) != len(fields):
                raise ValueError(f"wrong GQS cell count at line {line_number}")
            notes.append(dict(zip(fields, cells)))
    if metadata is None:
        raise ValueError("GQS has no metadata")
    return {"metadata": metadata, "notes": notes}
