"""Build the summary figure for the infrastructure–isolation note."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "experiments" / "theorem-network-measures"

rows = [
    json.loads(line)
    for line in (EXPERIMENT / "artifacts/scores.jsonl").read_text(encoding="utf-8").splitlines()
]
analysis = json.loads(
    (EXPERIMENT / "artifacts/analysis.json").read_text(encoding="utf-8")
)

reliance = np.asarray([row["dependency_mass_log"] for row in rows], dtype=np.float64)
statement = np.asarray(
    [row["statement_effective_neighbors"] for row in rows], dtype=np.float64
)
proof = np.asarray([row["proof_effective_neighbors"] for row in rows], dtype=np.float64)
complexity = np.asarray([row["structural_complexity"] for row in rows], dtype=np.float64)

if len(rows) != 10_000:
    raise RuntimeError("Expected the frozen 10,000-target score artifact")

# Equal-count reliance deciles, with deterministic tie handling inherited from
# NumPy's stable ordering of the frozen score array.
order = np.argsort(reliance, kind="stable")
decile = np.empty(len(rows), dtype=np.int32)
decile[order] = np.minimum(np.arange(len(rows)) * 10 // len(rows), 9)
x = np.arange(1, 11)
statement_median = np.asarray([np.median(statement[decile == d]) for d in range(10)])
proof_median = np.asarray([np.median(proof[decile == d]) for d in range(10)])
complexity_median = np.asarray([np.median(complexity[decile == d]) for d in range(10)])

matrix = np.column_stack([reliance, statement, proof, complexity])
correlation = np.asarray(spearmanr(matrix).statistic)

print(
    "records=10,000 "
    f"statement_endpoints={statement_median[0]:.2f},{statement_median[-1]:.2f} "
    f"proof_endpoints={proof_median[0]:.2f},{proof_median[-1]:.2f} "
    f"complexity_endpoints={complexity_median[0]:.3f},{complexity_median[-1]:.3f}"
)

mpl.rcParams.update(
    {
        "font.size": 8.2,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.edgecolor": "#aeb3c0",
        "axes.linewidth": 0.55,
        "grid.color": "#dfe2ea",
        "grid.linewidth": 0.55,
    }
)

violet = "#7454cc"
teal = "#168c8e"
orange = "#d77836"
rose = "#b63d6a"
ink = "#202332"

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 2.48),
    gridspec_kw={"width_ratios": [1.15, 0.95, 1.0], "wspace": 0.42},
)

axes[0].plot(x, statement_median, color=teal, marker="o", markersize=3.2, linewidth=1.6, label="statement")
axes[0].plot(x, proof_median, color=orange, marker="o", markersize=3.2, linewidth=1.6, label="proof")
axes[0].set_title("(a) Semantic neighborhoods", loc="left", pad=3)
axes[0].set_xlabel("dependency-reliance decile")
axes[0].set_ylabel("median effective neighbors")
axes[0].set_xticks([1, 3, 5, 7, 10])
axes[0].set_ylim(0, 70)
axes[0].grid(axis="y")
axes[0].legend(frameon=False, loc="upper right")
axes[0].annotate("64.5", (1, statement_median[0]), xytext=(3, 4), textcoords="offset points", color=teal, fontsize=7)
axes[0].annotate("1.9", (10, proof_median[-1]), xytext=(-2, 5), textcoords="offset points", ha="right", color=orange, fontsize=7)

axes[1].plot(x, complexity_median, color=rose, marker="o", markersize=3.2, linewidth=1.7)
axes[1].axhline(0, color="#8b91a0", linewidth=0.7, linestyle="--")
axes[1].fill_between(x, 0, complexity_median, color=rose, alpha=0.12)
axes[1].set_title("(b) Proof involvement", loc="left", pad=3)
axes[1].set_xlabel("dependency-reliance decile")
axes[1].set_ylabel("median complexity $C$")
axes[1].set_xticks([1, 3, 5, 7, 10])
axes[1].set_ylim(-1.15, 1.35)
axes[1].grid(axis="y")
axes[1].annotate("$-0.93$", (1, complexity_median[0]), xytext=(4, -1), textcoords="offset points", color=rose, fontsize=7)
axes[1].annotate("$1.20$", (10, complexity_median[-1]), xytext=(-3, 4), textcoords="offset points", ha="right", color=rose, fontsize=7)

cmap = LinearSegmentedColormap.from_list("principle", ["#39859c", "#ffffff", "#a63f72"])
image = axes[2].imshow(correlation, cmap=cmap, vmin=-1, vmax=1)
labels = ["$R$", "$S_S$", "$S_P$", "$C$"]
axes[2].set_xticks(range(4), labels)
axes[2].set_yticks(range(4), labels)
axes[2].set_title("(c) Rank correlations", loc="left", pad=3)
axes[2].grid(False)
for row in range(4):
    for column in range(4):
        value = correlation[row, column]
        color = "white" if abs(value) > 0.72 else ink
        axes[2].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.2, color=color)
for spine in axes[2].spines.values():
    spine.set_linewidth(0.45)
cbar = fig.colorbar(image, ax=axes[2], fraction=0.052, pad=0.04)
cbar.set_ticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=7, length=2)
cbar.outline.set_linewidth(0.4)

fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.22)
fig.savefig(HERE / "infrastructure_isolation.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "infrastructure_isolation.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
print("wrote infrastructure_isolation.pdf and .png")
