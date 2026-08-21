"""Build the compact proof-prefix mechanism figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
analysis = json.loads(
    (ROOT / "experiments/proof-prefix-trajectories/artifacts/analysis.json").read_text(
        encoding="utf-8"
    )
)
checkpoints = [1, 2, 4, 8, 16]


def medians(trajectory: str, metric: str) -> list[float]:
    return [
        analysis["checkpoint_summary"][trajectory][str(checkpoint)][metric]["median"]
        for checkpoint in checkpoints
    ]


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
orange = "#d77836"
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.68), gridspec_kw={"wspace": 0.38})

panels = [
    ("initial_neighbor_mean_similarity", "(a) Initial-community retention", "median cosine"),
    ("initial_neighbor_subspace_residual", "(b) Orthogonal residual", "median residual energy"),
    ("effective_neighbors", "(c) Global neighborhood", "median effective neighbors"),
]
for axis, (metric, title, ylabel) in zip(axes, panels):
    axis.plot(
        checkpoints, medians("actual", metric), color=violet, marker="o",
        linewidth=1.7, markersize=3.2, label="actual prefix",
    )
    axis.plot(
        checkpoints, medians("repeat", metric), color=orange, marker="o",
        linewidth=1.5, markersize=3.2, label="repeat first tactic",
    )
    axis.set_title(title, loc="left", pad=3)
    axis.set_xlabel("tactic steps")
    axis.set_ylabel(ylabel)
    axis.set_xticks(checkpoints)
    axis.grid(axis="y")

axes[0].set_ylim(0.55, 0.73)
axes[1].set_ylim(0.12, 0.28)
axes[2].set_ylim(0, 3.8)
axes[0].legend(frameon=False, loc="lower left")

fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.27)
fig.savefig(HERE / "proof_prefix_trajectories.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "proof_prefix_trajectories.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
print("wrote proof_prefix_trajectories.pdf and .png")
