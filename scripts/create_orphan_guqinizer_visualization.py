#!/usr/bin/env python3
"""Create a compact standalone viewer for orphan Guqinizer trajectories."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_text_jianzi_quality_filtered_minimal_final_v1/messages_train.jsonl"
DEFAULT_IDS = ROOT / "ABC_J/agent_training/source_quality_audit_v1/orphan_guqinizer_sample_ids.txt"
DEFAULT_AUDIT = ROOT / "ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_text_jianzi_quality_filtered_minimal_final_v1/teacher_trajectory_audit.jsonl"
DEFAULT_FRAGMENT = Path(r"C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/orphan-guqinizer-review.fragment.html")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT)
    args = parser.parse_args()
    wanted = set(line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip())
    audit_rows = {row.get("sample_id"): row for row in read_rows(args.audit)}
    data = []
    for row in read_rows(args.input):
        if row.get("sample_id") not in wanted:
            continue
        user = next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"), "")
        title = (re.search(r"^谱名｜(.+)$", user, re.M) or [None, ""])[1]
        phrase = (re.search(r"^当前段｜(.+)$", user, re.M) or [None, ""])[1]
        tuning = (re.search(r"^调弦｜(.+)$", user, re.M) or [None, ""])[1]
        messages = []
        for message in row.get("messages", []):
            messages.append({
                "role": message.get("role", ""),
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
                "tool_call_id": message.get("tool_call_id", ""),
                "name": message.get("name", ""),
            })
        audit = audit_rows.get(row["sample_id"], {})
        private = audit.get("teacher_private") or {}
        data.append({"sample_id": row["sample_id"], "title": title, "phrase": phrase, "tuning": tuning,
                     "annotation_gqs": private.get("annotation_gqs", ""), "messages": messages})
    data.sort(key=lambda item: item["sample_id"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    fragment = f'''<div id="orphan-review">
  <h2>孤立 Guqinizer 记录复核</h2>
  <p class="text-muted">共 {len(data)} 条；每条都有 Guqinizer，但缺少对应 Fingering。记录来自过滤前候选，最终候选已移除。</p>
  <div class="viz-controls">
    <label class="form-label" for="sample-select">选择记录</label>
    <select id="sample-select" class="form-select"></select>
  </div>
  <div id="record-summary" class="card" aria-live="polite"></div>
  <div id="annotation-panel"></div>
  <div id="message-list"></div>
  <script type="application/json" id="orphan-data">{payload}</script>
  <script>
  (() => {{
    const data = JSON.parse(document.getElementById('orphan-data').textContent);
    const select = document.getElementById('sample-select');
    const summary = document.getElementById('record-summary');
    const list = document.getElementById('message-list');
    const esc = (value) => String(value ?? '').replace(/[&<>]/g, (ch) => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[ch]));
    const pretty = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    data.forEach((item, index) => {{
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = item.sample_id;
      select.appendChild(option);
    }});
    function render(index) {{
      const item = data[index];
      summary.innerHTML = `<strong>${{esc(item.sample_id)}}</strong><br>曲名：${{esc(item.title)}}　当前段：${{esc(item.phrase)}}　调弦：${{esc(item.tuning)}}<br><span class="text-muted">孤立原因：候选中存在 Guqinizer 消息，但不存在同一 phrase 的 Fingering 消息。</span>`;
      document.getElementById('annotation-panel').innerHTML = `<section class="card"><h3>标注参考 GQS</h3><pre>${{esc(item.annotation_gqs || '（该记录没有嵌入标注 GQS）')}}</pre></section>`;
      list.innerHTML = item.messages.map((message, i) => {{
        const calls = message.tool_calls?.length ? `\\n\\n[tool_calls]\\n${{pretty(message.tool_calls)}}` : '';
        const meta = message.role === 'tool' ? ` · ${{esc(message.name || 'tool')}} · ${{esc(message.tool_call_id)}}` : '';
        return `<section class="card" aria-label="第 ${{i + 1}} 条 ${{esc(message.role)}} 消息"><h3>${{i + 1}}. ${{esc(message.role)}}${{meta}}</h3><pre>${{esc((message.content || '') + calls)}}</pre></section>`;
      }}).join('');
    }}
    select.addEventListener('change', () => render(Number(select.value)));
    if (data.length) render(0);
  }})();
  </script>
</div>'''
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(fragment, encoding="utf-8")
    print(json.dumps({"records": len(data), "fragment": str(args.fragment)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
