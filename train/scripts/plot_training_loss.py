#!/usr/bin/env python3
"""Plot Hugging Face Trainer loss records from a text log."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_records(path: Path) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        item = None
        # LLaMA-Factory's traditional trainer log is a Python dict repr.
        if line.startswith("{'loss':"):
            try:
                item = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                item = None
        else:
            # The current server monitor writes JSONL with current_steps.
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict):
                    item = candidate
            except json.JSONDecodeError:
                item = None
        if not isinstance(item, dict) or "loss" not in item:
            continue
        try:
            step = int(item.get("current_steps") or item.get("step") or 0)
            records.append((step, float(item["loss"])))
        except (ValueError, TypeError):
            continue
    if not records:
        return []
    # Older logs omit the step and are emitted every five optimizer steps.
    if all(step == 0 for step, _ in records):
        return [(5 * (i + 1), loss) for i, (_, loss) in enumerate(records)]
    # Keep the last record for a repeated step when a monitor appends a line
    # more than once during a retry/restart.
    deduped: dict[int, float] = {}
    for step, loss in records:
        deduped[step] = loss
    return sorted(deduped.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=1617)
    parser.add_argument("--title", default="训练 Loss 曲线")
    args = parser.parse_args()
    records = read_records(args.log)
    if not records:
        raise SystemExit("no Trainer loss records found")
    steps = [x for x, _ in records]
    loss = [y for _, y in records]
    window = min(11, len(loss))
    smooth = [sum(loss[i - window + 1 : i + 1]) / window for i in range(window - 1, len(loss))]
    smooth_steps = steps[window - 1 :]
    width, height = 1000, 560
    left, right, top, bottom = 78, 28, 54, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.25, max(0.8, max(loss) + 0.04)
    def xcoord(step: int) -> float:
        return left + step / args.total_steps * plot_w
    def ycoord(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h
    raw_points = " ".join(f"{xcoord(s):.1f},{ycoord(v):.1f}" for s, v in records)
    smooth_points = " ".join(f"{xcoord(s):.1f},{ycoord(v):.1f}" for s, v in zip(smooth_steps, smooth))
    grid = []
    for value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        if value < ymin or value > ymax:
            continue
        y = ycoord(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#cbd5e1" opacity=".45"/><text x="{left-12}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>')
    for step in (0, 250, 500, 750, 1000, 1250, 1500, args.total_steps):
        x = xcoord(step)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#cbd5e1" opacity=".25"/><text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle">{step}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{args.title}</title><desc id="desc">记录 loss 与 11 点移动平均，当前约 step {steps[-1]} / {args.total_steps}。</desc>
<rect width="100%" height="100%" fill="#f8fafc"/><text x="{left}" y="28" font-size="20" font-family="sans-serif" fill="#0f172a">{args.title}</text>
<g font-family="sans-serif" font-size="12" fill="#475569">{''.join(grid)}</g>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#64748b"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#64748b"/>
<polyline points="{raw_points}" fill="none" stroke="#4f83cc" stroke-width="1.4" opacity=".52"/><polyline points="{smooth_points}" fill="none" stroke="#d14d72" stroke-width="2.6"/>
<line x1="{xcoord(steps[-1]):.1f}" y1="{top}" x2="{xcoord(steps[-1]):.1f}" y2="{height-bottom}" stroke="#4f9b62" stroke-width="1.5" stroke-dasharray="6 5"/>
<text x="{left + plot_w/2:.1f}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">optimizer step</text><text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">training loss</text>
<g font-family="sans-serif" font-size="12" fill="#334155"><line x1="{width-300}" y1="28" x2="{width-270}" y2="28" stroke="#4f83cc" stroke-width="2"/><text x="{width-262}" y="32">记录 loss</text><line x1="{width-180}" y1="28" x2="{width-150}" y2="28" stroke="#d14d72" stroke-width="3"/><text x="{width-142}" y="32">{window} 点移动平均</text></g>
</svg>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"records={len(records)} current_step={steps[-1]} current_loss={loss[-1]:.4f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
