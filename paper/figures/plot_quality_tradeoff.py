"""Rebuild the paper's quality/cost figure from audited evaluation summaries.

Run from any directory: python /path/to/paper/figures/plot_quality_tradeoff.py
PDF is vector artwork; PNG is a review preview. No model/API calls are made.
"""
from collections import Counter, defaultdict
import math
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = [
    ("handbook", "GLM + Handbook", "#747C8C", "react_handbook_full_test50", "react_handbook_modern"),
    ("direct_sft", "Direct SFT", "#8062B2", "direct_sft_current48", "modern_direct_sft"),
    ("fixed_workflow", "Fixed workflow", "#CB8F38", "fixed_reference_ablation_current48", "modern_fixed_workflow"),
    ("single_stage", "GuqinAgent: single-stage", "#427DB4", "single_stage_current48", "modern_single_stage"),
    ("two_stage_off", "GuqinAgent: feedback OFF", "#B65D76", "warning_ablation_current48_off", "modern_two_stage_off"),
    ("two_stage", "GuqinAgent (Ours)", "#18847A", "warning_ablation_current48_on", "modern_two_stage"),
]


def read_json(path):
    return json.loads(path.read_text())


def diversity_adjusted_nonbasic_rate(directory):
    """Nonbasic usage discounted by within-score label repetition."""
    def event_key(event):
        return (event["piece_id"], event["phrase_id"], str(event["event_id"]))

    with (directory / "violations.jsonl").open() as stream:
        warned = {event_key(json.loads(line)) for line in stream}
    target_events = Counter()
    qualifying = Counter()
    techniques = defaultdict(Counter)
    with (directory / "per_event.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if not event.get("target_midi"):
                continue
            score = event["piece_id"]
            target_events[score] += 1
            labels = set(event["prediction"].get("ornaments") or ())
            if labels and event_key(event) not in warned:
                qualifying[score] += 1
                for label in sorted(labels):
                    techniques[score][label] += 1 / len(labels)
    weighted_sum = 0.0
    for score, count in qualifying.items():
        total = target_events[score]
        rate = count / total
        label_proportions = [mass / count for mass in techniques[score].values()]
        distinct = max(0.0, 1.0 - math.fsum(p * p for p in label_proportions))
        weighted_sum += total * rate * distinct
    all_targets = target_events.total()
    return weighted_sum / all_targets, qualifying.total() / all_targets


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B9C1CA", "axes.labelcolor": "#283540",
        "xtick.color": "#586673", "ytick.color": "#586673",
    })
    costs = read_json(OUT / "inference_cost.json")
    # One linear area scale shared by both datasets and all six methods.
    maximum_cost = max(r["estimated_usd_per_work_unit"] for r in costs["arms"].values())
    area_per_dollar = 180 / maximum_cost
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.05), sharex=True)
    fig.subplots_adjust(left=.16, right=.975, bottom=.11, top=.94, hspace=.39)
    offsets = [
        {"handbook": (65, 6), "fixed_workflow": (4, 15),
         "direct_sft": (45, 11), "single_stage": (53, 22),
         "two_stage_off": (39, 17), "two_stage": (74, 26)},
        {"handbook": (64, 7), "fixed_workflow": (4, 17),
         "direct_sft": (43, 14), "single_stage": (54, 28),
         "two_stage_off": (39, 23), "two_stage": (76, 18)},
    ]
    audit = []
    for panel, ax in enumerate(axes):
        ax.set(xlim=(0, 106), ylim=(0, 30), xticks=[0, 20, 40, 60, 80, 100],
               yticks=[0, 10, 20, 30],
               ylabel="Diversity-adjusted nonbasic rate (%)")
        ax.grid(color="#E7EBEF", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(["(a) Traditional: 48 scores", "(b) Modern: 10 melodies"][panel], loc="left", fontweight="bold", pad=5)
        for key, label, color, traditional, modern in METHODS:
            source = ROOT / "evaluation_results" / [traditional, modern][panel] / "summary.json"
            summary = read_json(source)
            metrics = summary["metrics"]
            if key == 'direct_sft':
                completion = summary['completion']
                assert completion['expected'] == [1031, 91][panel]
                assert completion['protocol_valid'] == [1003, 91][panel]
            else:
                assert summary["prediction_samples"] == [1031, 91][panel]
            x = 100 * metrics["pitch_accuracy_at_50_cents"]
            adjusted_rate, nonbasic_rate = diversity_adjusted_nonbasic_rate(source.parent)
            y = 100 * adjusted_rate
            assert math.isclose(nonbasic_rate, metrics["nonbasic_technique_rate"]["prediction_rate"])
            dataset = ["traditional", "modern"][panel]
            cost = costs["arms"][dataset + "/" + key]
            assert cost["experiment"] == [traditional, modern][panel]
            assert cost["samples"] == summary["prediction_samples"]
            assert hashlib.sha256((ROOT / cost["prediction_path"]).read_bytes()).hexdigest() == cost["prediction_sha256"]
            dollars_per_unit = cost["estimated_usd_per_work_unit"]
            ax.scatter(x, y, s=dollars_per_unit * area_per_dollar,
                       facecolor=color, edgecolor=color if key == 'direct_sft' else "white",
                       linewidth=.25 if key == 'direct_sft' else .6,
                       alpha=.80, zorder=4)
            caption = {"handbook": "GLM + Handbook", "fixed_workflow": "Fixed workflow",
                       "direct_sft": "Direct SFT" + ("‡" if panel == 0 else ""),
                       "single_stage": "Single-stage", "two_stage_off": "Feedback OFF",
                       "two_stage": "Ours"}[key]
            ax.annotate(caption, xy=(x, y), xytext=offsets[panel][key],
                        textcoords="data", fontsize=6, color=color,
                        fontweight="bold" if key == "two_stage" else "normal",
                        ha="left", va="center", zorder=5,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .75,
                                    "shrinkA": 2, "shrinkB": 1 if key == 'direct_sft' else 4},
                        bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1})
            audit.append({"panel": panel, "method": key, "source": str(source.relative_to(ROOT)),
                          "pitch_accuracy_percent": x, "diversity_adjusted_nonbasic_rate_percent": y,
                          "warning_free_nonbasic_rate_percent": 100 * nonbasic_rate,
                          "pitch_evaluable_events": metrics["pitch_evaluable_events"],
                          "estimated_api_usd_per_work_unit": dollars_per_unit,
                          "quality_scope": cost.get('quality_scope', 'complete test set'),
                          "cost_normalization": cost.get('normalization', 'per attempted work unit'),
                          "cost_scope": costs["usage_kind"]})
    axes[1].set_xlabel("Conditional pitch accuracy (%)")
    # Legend values are reference sizes, never per-point numeric annotations.
    legend_values = [maximum_cost / 8, maximum_cost / 2, maximum_cost]
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor="#8D9BA7",
                      markersize=(value * area_per_dollar)**.5, label=f"${100*value:.2f}")
               for value in legend_values]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .02),
                   title="API-equivalent USD / 100 units",
                   title_fontsize=5, ncol=3, frameon=True, framealpha=.95,
                   edgecolor="none", fontsize=5.5, handletextpad=.4, columnspacing=.8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "quality_tradeoff.pdf", bbox_inches="tight", pad_inches=.12)
    fig.savefig(OUT / "quality_tradeoff.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    (OUT / "quality_tradeoff_data.json").write_text(json.dumps(audit, indent=2) + "\n")
    table = ["% Generated by plot_quality_tradeoff.py from inference_cost.json.",
             r"\begin{tabularx}{\linewidth}{@{}Xrr@{}}", r"\toprule",
             r"Method & Traditional & Modern \\", r"\midrule"]
    for key, label, *_ in METHODS:
        amounts = [100 * costs["arms"][dataset + "/" + key]["estimated_usd_per_work_unit"]
                   for dataset in ("traditional", "modern")]
        name = label.replace("GuqinAgent: ", "")
        table.append(f"{name} & {amounts[0]:.3f} & {amounts[1]:.3f} " + r"\\")
    table.extend([r"\bottomrule", r"\end{tabularx}"])
    (OUT / "inference_cost_table.tex").write_text("\n".join(table) + "\n")
    print(OUT / "quality_tradeoff.pdf")


if __name__ == "__main__":
    main()
