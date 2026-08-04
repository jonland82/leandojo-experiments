"""Build the statement/proof comparison figure for the semantic note."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "experiments" / "semantic-embeddings-10000"

statement_embedding = np.load(
    EXPERIMENT / "artifacts" / "embeddings" / "statement.npy", allow_pickle=False
)
statement = np.load(
    EXPERIMENT / "artifacts" / "clusters" / "statement_labels.npy", allow_pickle=False
).astype(int)
proof = np.load(
    EXPERIMENT / "artifacts" / "clusters" / "proof_labels.npy", allow_pickle=False
).astype(int)
analysis = json.loads(
    (EXPERIMENT / "artifacts" / "analysis.json").read_text(encoding="utf-8")
)

n_statement = int(analysis["views"]["statement"]["selected_k"])
n_proof = int(analysis["views"]["proof"]["selected_k"])
if statement_embedding.shape != (len(statement), 1024) or len(statement) != len(proof):
    raise RuntimeError("semantic experiment artifacts have incompatible shapes")

# PCA is used only for the picture. Clustering and association statistics use
# the full normalized 1,024-dimensional vectors and all 10,000 records.
X_statement = normalize(statement_embedding.astype(np.float64), norm="l2")
projection = PCA(n_components=2, random_state=0).fit_transform(X_statement)

table = np.bincount(
    statement * n_proof + proof, minlength=n_statement * n_proof
).reshape(n_statement, n_proof)
row_share = table / table.sum(axis=1, keepdims=True)

print(
    f"proofs={len(statement)} "
    f"AMI={adjusted_mutual_info_score(statement, proof):.4f} "
    f"NMI={normalized_mutual_info_score(statement, proof):.4f} "
    f"max_row_share_min={row_share.max(axis=1).min():.4f} "
    f"max_row_share_median={np.median(row_share.max(axis=1)):.4f} "
    f"max_row_share_max={row_share.max(axis=1).max():.4f}"
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

tab20 = plt.get_cmap("tab20")
statement_cmap = ListedColormap([tab20(i) for i in range(n_statement)])
statement_norm = BoundaryNorm(
    np.arange(-0.5, n_statement + 0.5), statement_cmap.N
)
proof_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
proof_cmap = ListedColormap(proof_colors)
proof_norm = BoundaryNorm(np.arange(-0.5, n_proof + 0.5), proof_cmap.N)

display_rng = np.random.default_rng(0)
display_indices = np.sort(
    display_rng.choice(len(statement), size=min(3000, len(statement)), replace=False)
)
display_xy = projection[display_indices]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 2.5),
    gridspec_kw={"width_ratios": [1, 1, 0.88], "wspace": 0.32},
)

for ax, labels, cmap, norm, title in (
    (axes[0], statement, statement_cmap, statement_norm, "(a) Statement clusters"),
    (axes[1], proof, proof_cmap, proof_norm, "(b) Proof clusters"),
):
    display_labels = labels[display_indices]
    ax.scatter(
        display_xy[:, 0],
        display_xy[:, 1],
        s=6,
        c=display_labels,
        cmap=cmap,
        norm=norm,
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

heat = axes[2].imshow(row_share, cmap="Blues", vmin=0, vmax=0.7, aspect="auto")
axes[2].set_title("(c) Proof mix by statement", pad=3)
axes[2].set_xlabel("proof cluster", labelpad=2)
axes[2].set_ylabel("statement cluster", labelpad=2)
axes[2].set_xticks(range(n_proof), [str(i) for i in range(n_proof)])
axes[2].set_yticks(range(n_statement), [str(i) for i in range(n_statement)])
axes[2].tick_params(length=2, width=0.5, pad=1.5)
for spine in axes[2].spines.values():
    spine.set_color("#7f8792")
    spine.set_linewidth(0.5)

cbar = fig.colorbar(heat, ax=axes[2], fraction=0.052, pad=0.04)
cbar.set_label("row share", rotation=270, labelpad=9)
cbar.set_ticks([0, 0.2, 0.4, 0.6], labels=["0", ".2", ".4", ".6"])
cbar.outline.set_linewidth(0.4)

statement_handles = [
    mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=3.5,
        markerfacecolor=statement_cmap(i),
        markeredgewidth=0,
        label=f"S{i}",
    )
    for i in range(n_statement)
]
proof_handles = [
    mpl.lines.Line2D(
        [],
        [],
        marker="s",
        linestyle="",
        markersize=3.5,
        markerfacecolor=proof_colors[i],
        markeredgewidth=0,
        label=f"P{i}",
    )
    for i in range(n_proof)
]
fig.legend(
    handles=statement_handles + proof_handles,
    ncol=10,
    loc="lower left",
    bbox_to_anchor=(0.055, -0.01),
    frameon=False,
    handletextpad=0.15,
    columnspacing=0.55,
    borderaxespad=0,
    fontsize=7.0,
)

fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.20)
fig.savefig(HERE / "semantic_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(
    HERE / "semantic_comparison.png", dpi=240, bbox_inches="tight", pad_inches=0.02
)
print("wrote semantic_comparison.pdf and .png")
