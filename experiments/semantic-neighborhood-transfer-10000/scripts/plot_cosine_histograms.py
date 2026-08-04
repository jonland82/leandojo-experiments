"""Plot all-pairs cosine distributions for statement and proof embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
EMBEDDINGS = ROOT / "experiments" / "semantic-embeddings-10000" / "artifacts" / "embeddings"
OUTPUT = EXPERIMENT / "artifacts" / "cosine_similarity_histogram"
NORMAL_CHECK_OUTPUT = EXPERIMENT / "artifacts" / "cosine_similarity_normal_check"

N_BINS = 1_000
BLOCK_SIZE = 512


def normalized_embeddings(name: str) -> np.ndarray:
    vectors = np.load(EMBEDDINGS / f"{name}.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} embeddings contain a zero vector")
    return vectors / norms


def all_pair_histogram(vectors: np.ndarray, bins: np.ndarray, label: str) -> dict:
    """Accumulate similarities for every unordered pair without materializing n by n."""
    n = vectors.shape[0]
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total = 0
    total_similarity = 0.0
    total_squared_similarity = 0.0
    minimum = np.inf
    maximum = -np.inf

    starts = list(range(0, n, BLOCK_SIZE))
    for block_number, left_start in enumerate(starts, start=1):
        left_stop = min(left_start + BLOCK_SIZE, n)
        left = vectors[left_start:left_stop]
        for right_start in starts[block_number - 1 :]:
            right_stop = min(right_start + BLOCK_SIZE, n)
            similarities = left @ vectors[right_start:right_stop].T
            if right_start == left_start:
                upper = np.triu_indices(left_stop - left_start, k=1)
                values = similarities[upper]
            else:
                values = similarities.ravel()
            # Unit-vector dot products are cosines; remove float32 roundoff just
            # beyond the theoretical interval before binning.
            np.clip(values, -1.0, 1.0, out=values)

            counts += np.histogram(values, bins=bins)[0]
            total += values.size
            total_similarity += float(values.sum(dtype=np.float64))
            total_squared_similarity += float(
                np.square(values, dtype=np.float64).sum(dtype=np.float64)
            )
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))

        print(f"{label}: block {block_number}/{len(starts)}", flush=True)

    expected = n * (n - 1) // 2
    if total != expected or int(counts.sum()) != expected:
        raise RuntimeError(f"counted {total:,} pairs; expected {expected:,}")

    mean = total_similarity / total
    variance = max(total_squared_similarity / total - mean**2, 0.0)
    return {
        "counts": counts,
        "n_pairs": total,
        "mean": mean,
        "standard_deviation": variance**0.5,
        "minimum": minimum,
        "maximum": maximum,
    }


def histogram_quantiles(counts: np.ndarray, bins: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Approximate empirical quantiles by interpolating within fine histogram bins."""
    cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1])
    targets = probabilities * total
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.clip(indices, 0, len(counts) - 1)
    previous = np.where(indices == 0, 0, cumulative[indices - 1])
    within = np.divide(
        targets - previous,
        counts[indices],
        out=np.zeros_like(targets, dtype=np.float64),
        where=counts[indices] > 0,
    )
    return bins[indices] + within * (bins[indices + 1] - bins[indices])


def main() -> None:
    bins = np.linspace(-1.0, 1.0, N_BINS + 1)
    statement = all_pair_histogram(
        normalized_embeddings("statement"), bins, "statements"
    )
    proof = all_pair_histogram(normalized_embeddings("proof"), bins, "proofs")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )

    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    statement_density = statement["counts"] / (statement["n_pairs"] * width)
    proof_density = proof["counts"] / (proof["n_pairs"] * width)
    statement_color = "#4F738C"
    proof_color = "#D58B8B"

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.fill_between(
        centers,
        statement_density,
        step="mid",
        color=statement_color,
        alpha=0.30,
    )
    ax.step(
        centers,
        statement_density,
        where="mid",
        color=statement_color,
        linewidth=1.6,
        label=f"theorem statements  (mean {statement['mean']:.3f})",
    )
    ax.fill_between(
        centers,
        proof_density,
        step="mid",
        color=proof_color,
        alpha=0.26,
    )
    ax.step(
        centers,
        proof_density,
        where="mid",
        color=proof_color,
        linewidth=1.6,
        label=f"recorded proofs  (mean {proof['mean']:.3f})",
    )
    ax.axvline(statement["mean"], color=statement_color, linewidth=0.9, linestyle="--")
    ax.axvline(proof["mean"], color=proof_color, linewidth=0.9, linestyle="--")

    occupied = np.flatnonzero((statement["counts"] + proof["counts"]) > 0)
    lower = max(-1.0, bins[occupied[0]] - 0.025)
    upper = min(1.0, bins[occupied[-1] + 1] + 0.025)
    ax.set_xlim(lower, upper)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Pairwise similarity in statement and proof embedding spaces", pad=12)
    ax.text(
        0.5,
        1.01,
        f"All {statement['n_pairs']:,} distinct theorem pairs in each space",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555B61",
    )
    ax.grid(axis="y", color="#D7DCE1", linewidth=0.65)
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(colors="#222222", length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#9AA1A8")
        spine.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")

    # A separate diagnostic figure compares the empirical distributions with
    # moment-matched normal densities and standardized normal quantiles.
    diagnostic, diagnostic_axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    density_axis, qq_axis = diagnostic_axes
    x_grid = np.linspace(lower, upper, 800)

    for result, color, label, empirical_density in (
        (statement, statement_color, "theorem statements", statement_density),
        (proof, proof_color, "recorded proofs", proof_density),
    ):
        density_axis.step(
            centers,
            empirical_density,
            where="mid",
            color=color,
            linewidth=1.35,
            label=f"{label}: empirical",
        )
        density_axis.plot(
            x_grid,
            norm.pdf(x_grid, loc=result["mean"], scale=result["standard_deviation"]),
            color=color,
            linewidth=1.25,
            linestyle="--",
            label=f"{label}: fitted normal",
        )

    density_axis.set_xlim(lower, upper)
    density_axis.set_ylim(bottom=0)
    density_axis.set_xlabel("cosine similarity")
    density_axis.set_ylabel("density")
    density_axis.set_title("(a) Empirical density and normal fit", pad=8)
    density_axis.legend(frameon=False, fontsize=8, loc="upper left")

    probabilities = np.linspace(0.0001, 0.9999, 999)
    theoretical = norm.ppf(probabilities)
    for result, color, label in (
        (statement, statement_color, "theorem statements"),
        (proof, proof_color, "recorded proofs"),
    ):
        empirical = histogram_quantiles(result["counts"], bins, probabilities)
        standardized = (empirical - result["mean"]) / result["standard_deviation"]
        qq_axis.plot(
            theoretical,
            standardized,
            color=color,
            linewidth=1.45,
            label=label,
        )

    diagonal_min = float(theoretical.min())
    diagonal_max = float(theoretical.max())
    qq_axis.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="#555B61",
        linewidth=0.9,
        linestyle="--",
        label="exact normal",
    )
    qq_axis.set_xlim(diagonal_min, diagonal_max)
    qq_axis.set_ylim(diagonal_min, max(diagonal_max, qq_axis.get_ylim()[1]))
    qq_axis.set_xlabel("standard-normal quantile")
    qq_axis.set_ylabel("standardized empirical quantile")
    qq_axis.set_title("(b) Normal Q–Q check", pad=8)
    qq_axis.legend(frameon=False, fontsize=8, loc="upper left")

    for axis in diagnostic_axes:
        axis.grid(color="#D7DCE1", linewidth=0.65)
        axis.tick_params(colors="#222222", length=3, width=0.6)
        for spine in axis.spines.values():
            spine.set_color("#9AA1A8")
            spine.set_linewidth(0.6)

    diagnostic.suptitle("How Gaussian are pairwise cosine similarities?", fontsize=14, y=0.98)
    diagnostic.text(
        0.5,
        0.915,
        f"Moment-matched fits over all {statement['n_pairs']:,} distinct pairs per space",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555B61",
    )
    diagnostic.subplots_adjust(
        left=0.075, right=0.985, bottom=0.14, top=0.75, wspace=0.22
    )
    diagnostic.savefig(NORMAL_CHECK_OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    diagnostic.savefig(NORMAL_CHECK_OUTPUT.with_suffix(".pdf"), bbox_inches="tight")

    for label, result in (("statements", statement), ("proofs", proof)):
        print(
            f"{label}: n={result['n_pairs']:,}, mean={result['mean']:.6f}, "
            f"sd={result['standard_deviation']:.6f}, "
            f"range=[{result['minimum']:.6f}, {result['maximum']:.6f}]"
        )
    print(f"wrote {OUTPUT.with_suffix('.png')} and .pdf")
    print(f"wrote {NORMAL_CHECK_OUTPUT.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
