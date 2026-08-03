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

with (ROOT / "experiments" / "aws-10000" / "artifacts" / "proofs.json").open(
    encoding="utf-8"
) as f:
    payload = json.load(f)

points = payload["points"]
xy = np.asarray([p["tsne"][:2] for p in points])
style = np.asarray([p["c"] for p in points])
domain = np.asarray([p["domain_c"] for p in points])

paired = (style >= 0) & (domain >= 0)
n_style = int(payload["views"]["style"]["k"])
n_domain = int(payload["views"]["domain"]["k"])
table = np.zeros((n_domain, n_style), dtype=int)
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
        paired_domain * n_style + rng.permutation(paired_style),
        minlength=n_domain * n_style,
    ).reshape(n_domain, n_style)
    exceedances += np.sum((permuted - expected) ** 2 / expected) >= chi_squared

print(
    f"proofs={len(points)} displayed={min(3000, len(points))} paired={paired.sum()} "
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
n_topic_colors = max(n_style, n_domain)
topic_cmap = ListedColormap([tab10(i) for i in range(n_topic_colors)])
topic_norm = BoundaryNorm(np.arange(-0.5, n_topic_colors + 0.5), topic_cmap.N)

# The model and contingency statistics use all 10,000 proofs.  A fixed uniform
# subset keeps the two scatter panels legible while preserving their shared points.
display_rng = np.random.default_rng(0)
display_indices = np.sort(
    display_rng.choice(len(points), size=min(3000, len(points)), replace=False)
)
display_xy = xy[display_indices]

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
    display_labels = labels[display_indices]
    missing = display_labels < 0
    ax.scatter(
        display_xy[missing, 0],
        display_xy[missing, 1],
        s=4,
        c="#c7cbd1",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    shown = ~missing
    ax.scatter(
        display_xy[shown, 0],
        display_xy[shown, 1],
        s=6,
        c=display_labels[shown],
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
axes[2].set_xticks(range(n_style), [str(i) for i in range(n_style)])
axes[2].set_yticks(range(n_domain), [str(i) for i in range(n_domain)])
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
    for i in range(n_topic_colors)
]
fig.legend(
    handles=handles,
    title="topic id",
    ncol=n_topic_colors,
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
