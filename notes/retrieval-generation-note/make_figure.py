"""Build the results figure for the retrieval-generation companion note."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "experiments" / "retrieval-guided-proof-generation-100"

analysis = json.loads(
    (EXPERIMENT / "artifacts" / "analysis.json").read_text(encoding="utf-8")
)

conditions = ["no_retrieval", "random", "bm25", "semantic"]
labels = {
    "no_retrieval": "no retrieval",
    "random": "random",
    "bm25": "BM25",
    "semantic": "semantic",
}
colors = {
    "no_retrieval": "#777F87",
    "random": "#D58B8B",
    "bm25": "#91B8D0",
    "semantic": "#4F738C",
}
shared_color = "#D7DCE1"
baseline_only_color = "#D58B8B"

pass_rates = {
    condition: np.asarray(
        [
            analysis["condition_metrics"][condition]["pass_at_k"][str(k)]["rate"]
            for k in (1, 2, 3)
        ]
    )
    for condition in conditions
}

comparisons = ["no_retrieval", "random", "bm25"]
paired = [analysis["paired_pass_at_3"]["semantic_minus_" + baseline] for baseline in comparisons]
differences = np.asarray([row["difference"] for row in paired])
intervals = np.asarray([row["bootstrap_95"] for row in paired])
asymmetric_error = np.vstack([differences - intervals[:, 0], intervals[:, 1] - differences])

semantic_only = np.asarray([row["semantic_only"] for row in paired])
both = np.asarray([row["both_success"] for row in paired])
baseline_only = np.asarray([row["baseline_only"] for row in paired])

mpl.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 2.42),
    gridspec_kw={"width_ratios": [1.05, 0.95, 1.15], "wspace": 0.43},
)

# (a) Pass@k curves.
k_values = np.asarray([1, 2, 3])
for condition in conditions:
    axes[0].plot(
        k_values,
        100 * pass_rates[condition],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color=colors[condition],
    )
for condition in conditions:
    axes[0].text(
        3.08,
        100 * pass_rates[condition][-1],
        labels[condition],
        color=colors[condition],
        fontsize=7.2,
        va="center",
    )
axes[0].set_title("(a) Kernel-verified success", pad=4)
axes[0].set_xlabel("candidates allowed, $k$", labelpad=2)
axes[0].set_ylabel("pass@$k$ (percent)", labelpad=2)
axes[0].set_xticks(k_values)
axes[0].set_xlim(0.9, 3.78)
axes[0].set_ylim(8, 26)
axes[0].set_yticks([10, 15, 20, 25])
axes[0].grid(axis="y", color="#d9dde3", linewidth=0.55)

# (b) Paired semantic-minus-baseline differences.
y = np.arange(len(comparisons))
axes[1].axvline(0, color="#8F969D", linewidth=0.8, linestyle="--")
axes[1].errorbar(
    100 * differences,
    y,
    xerr=100 * asymmetric_error,
    fmt="o",
    markersize=5,
    color=colors["semantic"],
    ecolor=colors["semantic"],
    elinewidth=1.4,
    capsize=2.5,
)
axes[1].set_title("(b) Semantic gain at $k=3$", pad=4)
axes[1].set_xlabel("paired difference (points)", labelpad=2)
axes[1].set_yticks(y, [labels[name] for name in comparisons])
axes[1].set_xlim(-4, 19)
axes[1].set_xticks([0, 5, 10, 15])
axes[1].invert_yaxis()
axes[1].grid(axis="x", color="#d9dde3", linewidth=0.55)

# (c) Target-level success overlaps at pass@3.
y = np.arange(len(comparisons))
axes[2].barh(y, semantic_only, color=colors["semantic"], label="semantic only")
axes[2].barh(y, both, left=semantic_only, color=shared_color, label="both")
axes[2].barh(
    y,
    baseline_only,
    left=semantic_only + both,
    color=baseline_only_color,
    label="baseline only",
)
for row, (left, middle, right) in enumerate(zip(semantic_only, both, baseline_only)):
    for value, start in ((left, 0), (middle, left), (right, left + middle)):
        if value:
            axes[2].text(
                start + value / 2,
                row,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#111111",
            )
axes[2].set_title("(c) Discordant and shared wins", pad=4)
axes[2].set_xlabel("targets solved by either method", labelpad=2)
axes[2].set_yticks(y, [labels[name] for name in comparisons])
axes[2].invert_yaxis()
axes[2].set_xlim(0, 33)
axes[2].set_xticks([0, 10, 20, 30])
axes[2].grid(axis="x", color="#d9dde3", linewidth=0.55, zorder=0)

for ax in axes:
    ax.tick_params(length=2.5, width=0.5, pad=2, colors="#222222")
    for spine in ax.spines.values():
        spine.set_color("#9AA1A8")
        spine.set_linewidth(0.5)

fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.20)
fig.savefig(HERE / "retrieval_generation_results.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(
    HERE / "retrieval_generation_results.png",
    dpi=240,
    bbox_inches="tight",
    pad_inches=0.02,
)
print("wrote retrieval_generation_results.pdf and .png")
