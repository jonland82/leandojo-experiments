"""Build the synthesis figure for the proof-geometry literature note."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


feature = load_json("experiments/aws-10000/artifacts/cross_view.json")
semantic = load_json(
    "experiments/semantic-embeddings-10000/artifacts/analysis.json"
)
local = load_json(
    "experiments/semantic-neighborhood-transfer-10000/artifacts/analysis.json"
)
generation = load_json(
    "experiments/retrieval-guided-proof-generation-100/artifacts/analysis.json"
)

statement_proof = next(
    row
    for row in semantic["cross_view_associations"]
    if row["comparison"] == "statement vs proof"
)
k10 = next(row for row in local["neighborhood_results"] if row["k"] == 10)
pairwise = local["sampled_pair_results"]["correlations"]
pass3 = {
    condition: generation["condition_metrics"][condition]["pass_at_k"]["3"]["rate"]
    for condition in ("no_retrieval", "bm25", "semantic")
}

mpl.rcParams.update(
    {
        "font.size": 8.4,
        "axes.titlesize": 9.4,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    }
)

blue = "#4F738C"
light_blue = "#DCEAF2"
pale_blue = "#EDF4F7"
gray = "#D7DCE1"
dark_gray = "#5D6670"
light_red = "#E9C7C4"
black = "#17212A"

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.0, 2.78),
    gridspec_kw={"width_ratios": [1.0, 1.28], "wspace": 0.12},
)
for ax in axes:
    ax.set_axis_off()


def rounded(ax, xy, width, height, facecolor, edgecolor=dark_gray, radius=0.025):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=0.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    return patch


# (a) Three non-equivalent meanings of geometry.
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.98, "(a) Three meanings of proof geometry", va="top", weight="bold")

layers = [
    (
        0.70,
        "Intrinsic",
        "proof terms, paths, normalization",
        "Girard; homotopy type theory",
        pale_blue,
    ),
    (
        0.39,
        "Structural",
        "dependencies, modules, hyperedges",
        "Mathlib network; universal proof graph",
        light_blue,
    ),
    (
        0.08,
        "Learned",
        "vectors, cosine neighborhoods, topics",
        "formula/proof embeddings; retrieval",
        gray,
    ),
]
for y, title, objects, precedents, color in layers:
    rounded(ax, (0.08, y), 0.84, 0.20, color)
    ax.text(0.12, y + 0.145, title, fontsize=9.2, weight="bold", color=black)
    ax.text(0.12, y + 0.095, objects, fontsize=7.8, color=black)
    ax.text(0.12, y + 0.040, precedents, fontsize=7.0, color=dark_gray)

for y0, y1 in ((0.70, 0.59), (0.39, 0.28)):
    ax.add_patch(
        FancyArrowPatch(
            (0.50, y0),
            (0.50, y1),
            arrowstyle="<->",
            mutation_scale=8,
            linewidth=0.8,
            color=blue,
        )
    )
# (b) The three-note evidence chain, with exact values loaded from artifacts.
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.01, 0.98, "(b) What the three experiments add", va="top", weight="bold")

note_boxes = [
    (
        0.69,
        "I. Separate global views",
        "tactic style vs premise domain",
        f"AMI {feature['adjusted_mutual_information']:.3f}",
        pale_blue,
    ),
    (
        0.41,
        "II. A local bridge",
        "statement vs proof embeddings",
        f"AMI {statement_proof['ami']:.3f}; rank r {pairwise['spearman']:.3f}",
        light_blue,
    ),
    (
        0.13,
        "III. An operational test",
        "",
        f"kernel-checked pass@3: {100*pass3['semantic']:.0f}% semantic / {100*pass3['bm25']:.0f}% BM25 / {100*pass3['no_retrieval']:.0f}% none",
        gray,
    ),
]
for y, title, subtitle, metric, color in note_boxes:
    rounded(ax, (0.05, y), 0.90, 0.20, color)
    ax.text(0.08, y + 0.137, title, fontsize=8.0, weight="bold", color=black)
    ax.text(0.08, y + 0.055, subtitle, fontsize=7.0, color=dark_gray)
    metric_x = 0.08 if title.startswith("III") else 0.58
    ax.text(metric_x, y + 0.055, metric, fontsize=6.6, color=blue, weight="bold")

for y0, y1 in ((0.69, 0.62), (0.41, 0.34)):
    ax.add_patch(
        FancyArrowPatch(
            (0.50, y0),
            (0.50, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=blue,
        )
    )
ax.text(
    0.50,
    0.045,
    f"separation  $\u2192$  local coupling ($\u0394$ cosine {k10['proof_cosine_delta']:.3f})  $\u2192$  intervention",
    ha="center",
    fontsize=7.4,
    color=black,
    weight="bold",
)

fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.02)
fig.savefig(HERE / "proof_geometry_synthesis.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(
    HERE / "proof_geometry_synthesis.png",
    dpi=240,
    bbox_inches="tight",
    pad_inches=0.02,
)
print("wrote proof_geometry_synthesis.pdf and .png")
