#!/usr/bin/env python3
"""Pull a running single-score evaluation and render its current progress.

Each invocation downloads the current remote JSONL, combines it with the
local runtime input, and writes a self-contained HTML viewer.  The viewer is
deliberately tolerant of partial output: unfinished phrases are shown as
pending, while completed Base/Guqinizer traces remain expandable.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SSH_RUN = Path("D:/new_cip/ops/.ssh_run.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the production prompt builders so the viewer reflects the same public
# contract as the running evaluator rather than maintaining a second copy.
from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    blank_fingering_plan, public_system_for,
)
from agents.abc_to_jianzipu.teacher_trajectory import render_public_prompt  # noqa: E402
from scripts.audit_jianpu_jianzi_pitch import audit as audit_pitch  # noqa: E402


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pretty(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(SSH_RUN), "--get", remote, str(local)],
        check=True,
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def actions_by_index(stage: dict | None, field: str = "jianzi_text") -> dict[int, str]:
    actions = ((stage or {}).get("plan") or {}).get("actions") or []
    result: dict[int, str] = {}
    for action in actions:
        if not isinstance(action, dict) or action.get("source_index") is None:
            continue
        value = action.get(field)
        result[int(action["source_index"])] = "" if value is None else str(value)
    return result


def pitch_class(symbol: str) -> str:
    return {"✓": "matched", "✗": "mismatched"}.get(symbol, "unresolved")


def prompt_block(stage: str, system_prompt: str, user_prompt: str) -> str:
    """Render the exact public stage prompts used by the evaluator.

    The evaluator keeps prompts implicit in its deterministic loop, so the
    trace records do not contain them.  Reconstructing them here makes each
    phrase self-contained for inspection without exposing private references.
    """
    return (
        "<details class='prompt'><summary>模型输入 prompt（公开）</summary>"
        f"<h4>system｜{esc(stage)}</h4><pre>{esc(system_prompt)}</pre>"
        f"<h4>user｜{esc(stage)}</h4><pre>{esc(user_prompt)}</pre>"
        "</details>"
    )


def stage_trace(stage: dict | None, title: str, prompt_html: str = "") -> str:
    if not stage:
        return f"<details><summary>{esc(title)}：未生成</summary><p>该阶段尚未有结果。</p></details>"
    trace = stage.get("trace") or []
    sections = []
    for item in trace:
        round_no = item.get("round", "?")
        calls = item.get("tool_calls") or []
        results = item.get("tool_results") or []
        raw = item.get("raw_output") or ""
        suffix = "｜输出被截断" if item.get("truncated") else ""
        # ③ must show exactly what the model receives next round: the tool
        # RESULT only.  The record's name/arguments are audit provenance and
        # already visible in ②; echoing them here mislabels the input.
        result_blocks = "".join(
            f"<div class='round-step'><b>③ 工具返回（下一轮模型输入）｜{esc(entry.get('name') or '?')}</b>"
            f"<pre>{esc(pretty(entry.get('result')))}</pre></div>"
            for entry in results
        ) or "<div class='round-step'><b>③ 工具返回（下一轮模型输入）</b><pre>（本轮无工具执行）</pre></div>"
        sections.append(
            "<article class='round'>"
            f"<h4>第 {esc(round_no)} 轮<span>assistant → tool{esc(suffix)}</span></h4>"
            "<div class='round-step'><b>① 模型输出</b>"
            f"<pre>{esc(raw)}</pre></div>"
            "<div class='round-step'><b>② 解析出的工具调用</b>"
            f"<pre>{esc(pretty(calls))}</pre></div>"
            + result_blocks +
            "</article>"
        )
    status = "成功" if stage.get("ok") else "失败"
    if stage.get("no_op"):
        status += "｜no-op"
    return (
        f"<details><summary>{esc(title)}｜{status}｜{len(trace)} 轮</summary>"
        + prompt_html
        + "".join(sections)
        + "</details>"
    )


def stage_prompts(
    source: dict,
    runtime: dict,
    base: dict | None,
    previous_phrase: dict | None = None,
) -> tuple[str, str, str, str]:
    """Return Base and Guqinizer public system/user prompts for a phrase."""
    def with_previous(item: dict) -> dict:
        result = deepcopy(item)
        handoff = result.setdefault("input", {}).setdefault("phrase_handoff", {})
        handoff.pop("previous_phrase", None)
        if previous_phrase:
            handoff["previous_phrase"] = deepcopy(previous_phrase)
        return result

    # Base starts from the blank scaffold used by eval_two_stage_score.py.
    base_runtime = with_previous(runtime)
    base_runtime["baseline_plan"] = blank_fingering_plan(
        runtime.get("baseline_plan") or {"actions": []}
    )
    base_system = public_system_for("fingering_agent", basic=True)
    # Re-render with the preceding model-produced handoff.  The stored prompt
    # is retained as a fallback for older inputs that cannot be rendered.
    try:
        base_user = render_public_prompt(base_runtime, "fingering_agent")
    except Exception:
        base_user = str(source.get("public_prompt") or "")

    guqin_runtime = with_previous(runtime)
    if base and isinstance(base.get("plan"), dict):
        guqin_runtime["baseline_plan"] = deepcopy(base["plan"])
    else:
        guqin_runtime["baseline_plan"] = base_runtime["baseline_plan"]
    guqin_system = public_system_for("guqinization", basic=False)
    guqin_user = render_public_prompt(guqin_runtime, "guqinization")
    return base_system, base_user, guqin_system, guqin_user


def pitch_symbols(source: dict, actions: dict[int, str]) -> dict[int, str]:
    """Audit the final text against the phrase's normalized tuning.

    The symbols deliberately follow the full-trajectory viewer: ✓ is a
    resolved pitch match, ✗ is a mismatch, and ○ means the notation cannot be
    judged reliably (including display-only or non-sounding rows).
    """
    input_data = source.get("input") or {}
    notes = []
    for note in input_data.get("notes_without_jianzi") or []:
        if note.get("index") is None:
            continue
        row = dict(note)
        row["jianzi"] = actions.get(int(note["index"]), "")
        notes.append(row)
    try:
        report = audit_pitch({
            "metadata": dict(input_data.get("metadata") or {}),
            "open_midi": list((input_data.get("normalized_tuning") or {}).get("open_midi") or []),
            "notes": notes,
        }, 50.0)
    except Exception:
        return {int(note["index"]): "○" for note in notes}
    return {
        int(detail["index"]): (
            "✓" if detail.get("status") == "matched"
            else "✗" if detail.get("status") == "mismatched" else "○"
        )
        for detail in report.get("details") or []
        if detail.get("index") is not None
    }


def compare_table(source: dict, base: dict | None, guqinizer: dict | None, annotation: dict[int, str]) -> str:
    notes = ((source.get("input") or {}).get("notes_without_jianzi") or [])
    base_map = actions_by_index(base)
    gq_map = actions_by_index(guqinizer)
    base_pitch = pitch_symbols(source, base_map)
    gq_pitch = pitch_symbols(source, gq_map)
    # GQS 1.2: the model writes continuous 音序 (bars occupy no ordinal);
    # internal bookkeeping stays on source indexes, display uses 音序 like
    # visualize_all_trajectories_hierarchical.py.
    event_of = {
        int(note["index"]): note.get("event_index")
        for note in notes if note.get("index") is not None
    }
    indices = [int(note["index"]) for note in notes if note.get("index") is not None]
    indices = sorted(set(indices) | set(base_map) | set(gq_map))
    rows = []
    for index in indices:
        note = next((n for n in notes if int(n.get("index", -1)) == index), {})
        jianpu = note.get("jianpu") or ""
        abc = note.get("abc") or ""
        duration = note.get("duration") or ""
        event = event_of.get(index)
        ordinal = str(int(event)) if event is not None else "—"
        def shown(mapping: dict[int, str]) -> str:
            return mapping[index] if index in mapping and mapping[index] else "（空）"
        target = annotation.get(index, "") or "（空）"
        base_text = shown(base_map)
        gq_text = shown(gq_map)
        base_class = "same" if base_text == target else "diff"
        gq_class = "same" if gq_text == target else "diff"
        rows.append(
            "<tr>"
            f"<th title='source index {index}'>{ordinal}</th><td>{esc(jianpu)}</td><td>{esc(abc)}</td><td>{esc(duration)}</td>"
            f"<td class='{base_class}'>{esc(base_text)}</td>"
            f"<td class='{gq_class}'>{esc(gq_text)}</td><td class='annotation'>{esc(target)}</td>"
            f"<td class='pitch {pitch_class(base_pitch.get(index, '○'))}' title='✓ 匹配｜✗ 不匹配｜○ 无法可靠解析'>{base_pitch.get(index, '○')}</td>"
            f"<td class='pitch {pitch_class(gq_pitch.get(index, '○'))}' title='✓ 匹配｜✗ 不匹配｜○ 无法可靠解析'>{gq_pitch.get(index, '○')}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>序号</th><th>简谱</th><th>ABC</th><th>时值</th>"
            "<th>Base/Fingering 最终</th><th>Guqinizer 当前最终</th><th>标注</th>"
            "<th>Fingering 音高</th><th>Guqinizer 音高</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def notation_overview(source_rows: list[dict], predictions: dict[str, dict]) -> str:
    """Render each phrase in wrapped jianpu/jianzi pairs.

    A complete phrase can contain enough verbose jianzi that a single
    horizontal grid becomes impractical to inspect.  Keep the two notation
    rows aligned, but split every phrase into small, self-contained strips.
    """
    notes_per_strip = 8
    lines = []
    for source in source_rows:
        sample_id = str(source.get("sample_id"))
        runtime = source.get("runtime_item") or source
        notes = ((runtime.get("input") or {}).get("notes_without_jianzi") or [])
        prediction = predictions.get(sample_id) or {}
        final_stage = prediction.get("guqinizer") or {}
        label = "Guqinizer"
        if not isinstance(final_stage.get("plan"), dict):
            final_stage = prediction.get("base") or {}
            label = "Base" if sample_id in predictions else "待生成"
        actions = actions_by_index(final_stage)
        cells: list[tuple[str, str]] = []
        for note in notes:
            if note.get("index") is None:
                continue
            index = int(note["index"])
            jianpu = str(note.get("jianpu") or note.get("jianpu_alt") or "—")
            if str(note.get("abc") or "").strip() == "|":
                jianpu = "｜"
            jianzi = actions.get(index)
            event = note.get("event_index")
            ordinal = str(int(event)) if event is not None else "—"
            cells.append((
                f"<div class='notation-cell' title='source index {index}'><small>{ordinal}</small>{esc(jianpu)}</div>",
                f"<div class='notation-cell jianzi'>{esc(jianzi if jianzi else '·')}</div>",
            ))
        phrase_id = runtime.get("phrase_id") or source.get("phrase_id") or sample_id
        strip_count = max(1, (len(cells) + notes_per_strip - 1) // notes_per_strip)
        strips = []
        for start in range(0, len(cells), notes_per_strip):
            chunk = cells[start:start + notes_per_strip]
            strip_index = start // notes_per_strip + 1
            strip_label = (
                f"{phrase_id}｜第 {strip_index}/{strip_count} 行"
                if strip_count > 1 else str(phrase_id)
            )
            strips.append(
                "<section class='notation-strip'>"
                f"<header><code>{esc(strip_label)}</code><span>{esc(label)}</span></header>"
                "<div class='notation-grid' "
                f"style='--note-count:{len(chunk)}'>"
                "<div class='notation-label'>简谱</div>" + "".join(item[0] for item in chunk)
                + "<div class='notation-label'>减字</div>" + "".join(item[1] for item in chunk)
                + "</div></section>"
            )
        lines.append("<section class='notation-line'>" + "".join(strips) + "</section>")
    return (
        "<section class='notation-overview'><h2>曲谱总览</h2>"
        "<p>每段上为简谱，下为当前模型最终减字；较长段每 8 音自动换行；“·”表示尚未提交或该行为空。</p>"
        + "".join(lines) + "</section>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", default="ScGgTmDm")
    parser.add_argument("--remote", default="/home/20223393ljw/guqin-agent/train/eval_outputs_v3_two_stage/ScGgTmDm.jsonl")
    # The local test.jsonl is the runtime-enriched input in this checkout;
    # older server snapshots call the same format test_runtime.jsonl.
    parser.add_argument("--input", type=Path, default=ROOT / "train/eval_inputs_v2_text_protocol/test.jsonl")
    parser.add_argument("--references", type=Path, default=ROOT / "ABC_J/agent_training/reference_trajectories_test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "train/eval_outputs_v3_two_stage/ScGgTmDm_running_view.html")
    parser.add_argument("--pulled", type=Path, default=ROOT / "train/eval_outputs_v3_two_stage/ScGgTmDm.remote.jsonl")
    parser.add_argument(
        "--focus-sample",
        help="sample_id to expand by default (for focused trajectory review)",
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="render an already downloaded --pulled JSONL without opening an SSH connection",
    )
    args = parser.parse_args()
    if not args.skip_pull:
        pull(args.remote, args.pulled)

    source_rows = [r for r in load_jsonl(args.input) if str(r.get("score_key")) == args.score]
    predictions = {
        str(r.get("sample_id")): r
        for r in load_jsonl(args.pulled)
        if str(r.get("score_key")) == args.score
    }
    source_by_id = {str(r.get("sample_id")): r for r in source_rows}
    reference_by_id: dict[str, dict[int, str]] = {}
    for row in load_jsonl(args.references):
        if str(row.get("score_key")) != args.score:
            continue
        ref_map: dict[int, str] = {}
        # The older trajectory file carries reference_plan.actions; the
        # evaluation-pairs manifest carries sealed reference.actions.  Both
        # represent the same source-indexed notation for visualization.
        actions = ((row.get("reference_plan") or {}).get("actions") or [])
        if not actions:
            actions = ((row.get("reference") or {}).get("actions") or [])
        for action in actions:
            if isinstance(action, dict) and action.get("source_index") is not None:
                ref_map[int(action["source_index"])] = str(
                    action.get("jianzi") or action.get("jianzi_text") or ""
                )
        reference_by_id[str(row.get("score_key")) + "-" + str(row.get("phrase_id"))] = ref_map
    ordered_ids = [str(r.get("sample_id")) for r in source_rows]
    completed = sum(sample_id in predictions for sample_id in ordered_ids)
    title = ""
    blocks = []
    previous_for_prompt: dict | None = None
    previous_score = None
    for position, sample_id in enumerate(ordered_ids):
        source = source_by_id[sample_id]
        runtime = source.get("runtime_item") or source
        score_key = str(source.get("score_key") or runtime.get("score_key") or "")
        if previous_score is not None and score_key != previous_score:
            previous_for_prompt = None
        previous_score = score_key
        title = title or ((runtime.get("input") or {}).get("metadata") or {}).get("score_title", "")
        prediction = predictions.get(sample_id)
        base = (prediction or {}).get("base") if prediction else None
        guqinizer = (prediction or {}).get("guqinizer") if prediction else None
        annotation = reference_by_id.get(sample_id, {})
        status = "已完成" if prediction else "等待输出"
        if prediction and not prediction.get("protocol_valid"):
            status = "阶段失败"
        try:
            base_system, base_user, guqin_system, guqin_user = stage_prompts(
                source, runtime, base, previous_for_prompt
            )
            base_prompt_html = prompt_block(
                "fingering_agent", base_system, base_user
            )
            guqin_prompt_html = prompt_block(
                "guqinization", guqin_system, guqin_user
            )
        except Exception as exc:
            # Keep partially written/legacy inputs viewable even if one prompt
            # cannot be reconstructed; the trace and comparison remain useful.
            error = f"prompt reconstruction failed: {type(exc).__name__}: {exc}"
            base_prompt_html = f"<p>{esc(error)}</p>"
            guqin_prompt_html = f"<p>{esc(error)}</p>"
        blocks.append(
            f"<details class='phrase'{' open' if (sample_id == args.focus_sample or (not args.focus_sample and position == max(0, completed - 1))) else ''}>"
            f"<summary><code>{esc(sample_id)}</code>｜{status}</summary>"
            f"<p>当前段：{esc(source.get('phrase_id'))}；这是评估运行时输入，不含私有标注。</p>"
            f"<h3>当前结果 vs 标注</h3>{compare_table(runtime, base, guqinizer, annotation)}"
            f"<h3>两阶段轨迹</h3>{stage_trace(base, 'Base/Fingering', base_prompt_html)}"
            f"{stage_trace(guqinizer, 'Guqinizer', guqin_prompt_html)}"
            f"<details><summary>原始评估记录</summary><pre>{esc(pretty(prediction))}</pre></details>"
            "</details>"
        )
        if prediction and prediction.get("protocol_valid"):
            final_stage = (prediction.get("guqinizer") or {}).get("plan")
            if not isinstance(final_stage, dict):
                final_stage = (prediction.get("base") or {}).get("plan")
            if isinstance(final_stage, dict):
                previous_for_prompt = {
                    "phrase_id": runtime.get("phrase_id"),
                    "status": "confirmed_readonly",
                    "notes": deepcopy((runtime.get("input") or {}).get(
                        "notes_without_jianzi", []
                    )),
                    "actions": deepcopy(final_stage.get("actions") or []),
                }
    overview = notation_overview(source_rows, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{esc(args.score)}｜运行中评估</title><style>
:root{{--bg:#181818;--fg:#e8e4dc;--line:#494949;--surface:#252525;--muted:#b9b3a8;--accent:#2f3a40;--same:#1e3a26;--diff:#40261e}}
*{{box-sizing:border-box}}body{{max-width:1380px;margin:24px auto;padding:0 16px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif}}h1{{font-size:23px;margin:0 0 8px}}h2,h3,h4{{font-weight:500;margin:14px 0 8px}}p{{color:var(--muted)}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface);overflow:hidden}}summary{{cursor:pointer;padding:9px 12px;font-weight:500;background:var(--accent);color:var(--fg)}}.phrase>summary{{background:#2c2c2c}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#1d1d1d;color:#e8e4dc;font:12px/1.45 ui-monospace,'Cascadia Code',monospace}}.table-wrap{{overflow:auto;margin:8px 0 12px}}table{{border-collapse:collapse;width:100%;min-width:1080px;background:var(--surface)}}th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}}thead th{{font-weight:500;background:#303030;position:sticky;top:0}}tbody th{{font-weight:500;white-space:nowrap}}code{{font-family:ui-monospace,'Cascadia Code',monospace}}.status{{padding:10px 12px;background:#33301c;border:1px solid #6b6237;border-radius:8px;color:var(--fg)}}.pitch{{width:82px;text-align:center;font-size:18px;font-weight:700}}.pitch.matched{{color:#7dd87d;background:#1e3a26}}.pitch.mismatched{{color:#ef8a7a;background:#40261e}}.pitch.unresolved{{color:var(--muted);background:#2e2c29}}
</style><style>
.notation-overview{{margin:20px 0}}.notation-line{{margin:10px 0;background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden}}.notation-strip+.notation-strip{{border-top:2px solid var(--line)}}.notation-strip header{{display:flex;gap:10px;justify-content:space-between;padding:7px 10px;background:#303030;color:var(--muted)}}.notation-grid{{display:grid;grid-template-columns:60px repeat(var(--note-count),minmax(0,1fr));width:100%}}.notation-label{{padding:8px;background:var(--accent);font-weight:500;border-bottom:1px solid var(--line)}}.notation-cell{{min-width:0;min-height:42px;padding:5px 7px;border-left:1px solid var(--line);border-bottom:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere}}.notation-cell small{{display:block;color:var(--muted);font:11px ui-monospace,monospace}}.notation-cell.jianzi{{min-height:50px;background:#202020}}.round{{margin:12px 0;border:1px solid var(--line);border-radius:8px;overflow:hidden}}.round h4{{display:flex;justify-content:space-between;gap:12px;margin:0;padding:8px 10px;background:#2c2c2c}}.round h4 span{{color:var(--muted);font-weight:400}}.round-step{{border-top:1px solid var(--line)}}.round-step b{{display:block;padding:6px 10px;background:var(--accent);font-weight:500}}
</style></head><body><h1>{esc(args.score)}｜{esc(title)}｜当前评估状态</h1>
<div class='status'>已拉取远端结果：{completed} / {len(ordered_ids)} 个 phrase。重新运行本脚本即可刷新；表格包含本地标注，仅用于检查，不会发送到服务器。</div>
{overview}
{''.join(blocks)}
</body></html>"""
    args.output.write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"pulled": str(args.pulled), "output": str(args.output), "completed": completed, "total": len(ordered_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
