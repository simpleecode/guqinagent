"""Export private teacher request/response traces as readable UTF-8 text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def readable(value: object) -> str:
    if value is None:
        return "null"
    text = str(value)
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace('\\"', '"')


def pretty(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    text = readable(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    return readable(json.dumps(parsed, ensure_ascii=False, indent=2))


def render(record: dict) -> str:
    lines = [
        "教师模型输入输出｜状态｜已读",
        f"样本ID｜{record.get('sample_id', '')}",
        "说明｜这里是教师模型的私有 request/response trace；转义换行已展开。",
        "",
    ]
    traces = (record.get("teacher_private") or {}).get("teacher_io_trace") or []
    for trace in traces:
        lines.append(f"【教师回合 {trace.get('round', '')}】")
        request = trace.get("request") or {}
        lines.append("【request】")
        for key in ("model", "max_tokens", "temperature", "thinking"):
            if key in request:
                lines.append(f"{key}｜{pretty(request[key])}")
        if "system" in request:
            lines.append("system｜")
            lines.append(pretty(request["system"]))
        for message in request.get("messages") or []:
            lines.append(f"message[{message.get('role', '')}]｜")
            lines.append(pretty(message.get("content")))
        response = trace.get("response") or {}
        lines.append("【response】")
        for key in ("model", "role", "stop_reason", "thinking"):
            if key in response:
                lines.append(f"{key}｜{pretty(response[key])}")
        if "content" in response:
            lines.append("content｜")
            lines.append(pretty(response["content"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("sample_ids", nargs="+")
    args = parser.parse_args()
    wanted = set(args.sample_ids)
    found: dict[str, dict] = {}
    with args.audit.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                if row.get("sample_id") in wanted:
                    found[row["sample_id"]] = row
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"missing sample_id: {', '.join(sorted(missing))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, record in found.items():
        path = args.output_dir / f"{sample_id}.txt"
        path.write_text(render(record), encoding="utf-8", newline="\n")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
