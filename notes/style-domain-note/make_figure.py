"""Build the comparison figure for the style/domain note."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

with (ROOT / "out" / "proofs.json").open(encoding="utf-8") as f:
    payload = json.load(f)

points = payload["points"]
xy = np.asarray([p["tsne"][:2] for p in points])
style = np.asarray([p["c"] for p in points])
domain = np.asarray([p["domain_c"] for p in points])

paired = (style >= 0) & (domain >= 0)
table = np.zeros((10, 10), dtype=int)
for d, s in zip(domain[paired], style[paired]):
    table[d, s] += 1
row_share = table / table.sum(axis=1, keepdims=True)
expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / table.sum()
chi_squared = float(np.sum((table - expected) ** 2 / expected))
cramers_v = float(
    np.sqrt(chi_squared / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1)))
)

# A fixed-margin label-permutation check avoids leaning on the asymptotic
# chi-squared p-value for the few cells with small expected counts.
rng = np.random.default_rng(0)
n_permutations = 20_000
exceedances = 0
paired_style = style[paired]
paired_domain = domain[paired]
for _ in range(n_permutations):
    permuted = np.bincount(
        paired_domain * 10 + rng.permutation(paired_style), minlength=100
    ).reshape(10, 10)
    exceedances += np.sum((permuted - expected) ** 2 / expected) >= chi_squared

print(
    f"paired={paired.sum()} "
    f"AMI={adjusted_mutual_info_score(paired_style, paired_domain):.4f} "
    f"NMI={normalized_mutual_info_score(paired_style, paired_domain):.4f} "
    f"chi2={chi_squared:.2f} CramersV={cramers_v:.3f} "
    f"permutation_exceedances={exceedances}/{n_permutations}"
)

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

tab10 = plt.get_cmap("tab10")
topic_cmap = ListedColormap([tab10(i) for i in range(10)])
topic_norm = BoundaryNorm(np.arange(-0.5, 10.5), topic_cmap.N)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 2.5),
    gridspec_kw={"width_ratios": [1, 1, 1.08], "wspace": 0.32},
)

for ax, labels, title in (
    (axes[0], style, "(a) Dominant style topic"),
    (axes[1], domain, "(b) Dominant domain topic"),
):
    missing = labels < 0
    ax.scatter(
        xy[missing, 0],
        xy[missing, 1],
        s=4,
        c="#c7cbd1",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    shown = ~missing
    ax.scatter(
        xy[shown, 0],
        xy[shown, 1],
        s=6,
        c=labels[shown],
        cmap=topic_cmap,
        norm=topic_norm,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b8bdc6")
        spine.set_linewidth(0.5)

heat = axes[2].imshow(row_share, cmap="Blues", vmin=0, vmax=0.32, aspect="auto")
axes[2].set_title("(c) Style mix within each domain", pad=3)
axes[2].set_xlabel("style topic $S_j$", labelpad=2)
axes[2].set_ylabel("domain topic $D_i$", labelpad=2)
axes[2].set_xticks(range(10), [str(i) for i in range(10)])
axes[2].set_yticks(range(10), [str(i) for i in range(10)])
axes[2].tick_params(length=2, width=0.5, pad=1.5)
for spine in axes[2].spines.values():
    spine.set_color("#7f8792")
    spine.set_linewidth(0.5)

cbar = fig.colorbar(heat, ax=axes[2], fraction=0.047, pad=0.03)
cbar.set_label("row share", rotation=270, labelpad=9)
cbar.set_ticks([0, 0.1, 0.2, 0.3], labels=["0", ".1", ".2", ".3"])
cbar.outline.set_linewidth(0.4)

# A compact categorical key shared by the two scatter panels.
handles = [
    mpl.lines.Line2D(
        [], [], marker="o", linestyle="", markersize=4,
        markerfacecolor=tab10(i), markeredgewidth=0, label=str(i)
    )
    for i in range(10)
]
fig.legend(
    handles=handles,
    title="topic id",
    ncol=10,
    loc="lower left",
    bbox_to_anchor=(0.06, -0.015),
    frameon=False,
    handletextpad=0.2,
    columnspacing=0.65,
    borderaxespad=0,
    fontsize=7.5,
    title_fontsize=7.5,
)

fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.15)
fig.savefig(HERE / "style_domain_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "style_domain_comparison.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
print("wrote style_domain_comparison.pdf and .png")
