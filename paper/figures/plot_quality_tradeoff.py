"""Rebuild the paper's quality/cost figure from audited evaluation summaries.

Run from any directory: python /path/to/paper/figures/plot_quality_tradeoff.py
PDF is vector artwork; PNG is a review preview. No model/API calls are made.
"""
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
    ("fixed_workflow", "Fixed workflow", "#CB8F38", "fixed_reference_ablation_current48", "modern_fixed_workflow"),
    ("single_stage", "GuqinAgent: single-stage", "#427DB4", "single_stage_current48", "modern_single_stage"),
    ("two_stage_off", "GuqinAgent: feedback OFF", "#B65D76", "warning_ablation_current48_off", "modern_two_stage_off"),
    ("two_stage", "GuqinAgent (Ours)", "#18847A", "warning_ablation_current48_on", "modern_two_stage"),
]


def read_json(path):
    return json.loads(path.read_text())


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
    # One linear area scale shared by both datasets and all five methods.
    maximum_cost = max(r["estimated_usd_per_work_unit"] for r in costs["arms"].values())
    area_per_dollar = 180 / maximum_cost
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.05), sharex=True)
    fig.subplots_adjust(left=.16, right=.975, bottom=.11, top=.94, hspace=.39)
    offsets = [
        {"handbook": (66, 6), "fixed_workflow": (6, 33),
         "single_stage": (57, 31), "two_stage_off": (44, 15), "two_stage": (68, 19)},
        {"handbook": (66, 23), "fixed_workflow": (3, 51),
         "single_stage": (57, 53), "two_stage_off": (37, 39), "two_stage": (68, 35)},
    ]
    audit = []
    for panel, ax in enumerate(axes):
        ax.set(xlim=(0, 106), ylim=(0, [40, 58][panel]), xticks=[0, 20, 40, 60, 80, 100],
               yticks=[0, 20, 40], ylabel="Ornament F1 (%)")
        ax.grid(color="#E7EBEF", linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(["(a) Traditional: 48 scores", "(b) Modern: 10 melodies"][panel], loc="left", fontweight="bold", pad=5)
        for key, label, color, traditional, modern in METHODS:
            source = ROOT / "evaluation_results" / [traditional, modern][panel] / "summary.json"
            summary = read_json(source)
            metrics = summary["metrics"]
            assert summary["prediction_samples"] == [1031, 91][panel]
            x = 100 * metrics["pitch_accuracy_at_50_cents"]
            y = 100 * metrics["ornament"]["f1"]
            dataset = ["traditional", "modern"][panel]
            cost = costs["arms"][dataset + "/" + key]
            assert cost["experiment"] == [traditional, modern][panel]
            assert cost["samples"] == summary["prediction_samples"]
            assert hashlib.sha256((ROOT / cost["prediction_path"]).read_bytes()).hexdigest() == cost["prediction_sha256"]
            dollars_per_unit = cost["estimated_usd_per_work_unit"]
            ax.scatter(x, y, s=dollars_per_unit * area_per_dollar,
                       facecolor=color, edgecolor="white", linewidth=.6,
                       alpha=.80, zorder=4)
            caption = {"handbook": "GLM + Handbook", "fixed_workflow": "Fixed workflow",
                       "single_stage": "Single-stage", "two_stage_off": "Feedback OFF",
                       "two_stage": "Ours"}[key]
            ax.annotate(caption, xy=(x, y), xytext=offsets[panel][key],
                        textcoords="data", fontsize=6, color=color,
                        fontweight="bold" if key == "two_stage" else "normal",
                        ha="left", va="center", zorder=5,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .75,
                                    "shrinkA": 2, "shrinkB": 4},
                        bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1})
            audit.append({"panel": panel, "method": key, "source": str(source.relative_to(ROOT)),
                          "pitch_accuracy_percent": x, "ornament_f1_percent": y,
                          "pitch_evaluable_events": metrics["pitch_evaluable_events"],
                          "estimated_api_usd_per_work_unit": dollars_per_unit,
                          "cost_scope": costs["usage_kind"]})
    axes[1].set_xlabel("Conditional pitch accuracy (%)")
    # Legend values are reference sizes, never per-point numeric annotations.
    legend_values = [maximum_cost / 8, maximum_cost / 2, maximum_cost]
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor="#8D9BA7",
                      markersize=(value * area_per_dollar)**.5, label=f"${100*value:.2f}")
               for value in legend_values]
    axes[1].legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .02),
                   title="API-equivalent USD / 100 units\n(no cache discount)",
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
