"""Generate the two summary figures used by the short paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def model_comparison() -> None:
    labels = [
        r"Matter-like ($p=2/3$)",
        r"Coasting ($p=1$)",
        r"3D + time ($p=4/3$)",
        r"Best constant $p$",
        r"Two channel",
        r"Flat $\Lambda$CDM",
    ]
    chi2 = np.array([1420.18, 187.09, 1364.93, 96.97, 10.27, 10.63])
    dof = np.array([12, 12, 12, 11, 11, 12])
    reduced = chi2 / dof
    colors = ["#9CA3AF", "#6B7280", "#9CA3AF", "#4B5563", "#2A9D8F", "#264653"]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, reduced, color=colors, height=0.66)
    ax.axvline(1.0, color="#B45309", linestyle="--", linewidth=1.2, label=r"$\chi^2/\nu=1$")
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Reduced $\chi^2$ for DESI DR2 BAO shape")
    ax.set_title("Constant-dimensional histories fail the distance--redshift shape")
    ax.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, reduced):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_xlim(0.65, 180)
    fig.tight_layout()
    fig.savefig(FIGURES / "desi_model_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "desi_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def neighborhood_control() -> None:
    # Means and sample standard deviations across realization--interval results.
    panels = [
        (
            "Fixed-cosmology hydro (10 tests)",
            np.array([[3.79974e-5, 1.32934e-6], [8.39086e-4, 1.40251e-4]]),
            np.array([[8.52124e-6, 3.17273e-7], [1.52878e-4, 6.38529e-5]]),
        ),
        (
            "Matched hydro (6 tests)",
            np.array([[5.27815e-5, 2.44625e-6], [1.04096e-3, 2.31970e-4]]),
            np.array([[2.13238e-5, 2.03289e-6], [4.64102e-5, 1.34400e-4]]),
        ),
        (
            "Matched gravity only (6 tests)",
            np.array([[5.49077e-5, 2.05699e-6], [1.24076e-3, 3.20891e-4]]),
            np.array([[2.11425e-5, 1.32361e-6], [1.15818e-4, 5.60206e-5]]),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.55), sharey=True)
    x = np.arange(2)
    width = 0.34
    for ax, (title, values, errors) in zip(axes, panels):
        for offset, state, color in (
            (-width / 2, 0, "#457B9D"),
            (width / 2, 1, "#E76F51"),
        ):
            ax.bar(
                x + offset,
                values[:, state],
                width,
                yerr=errors[:, state],
                capsize=2.5,
                color=color,
                label=("Raw change" if state == 0 else "After velocity conditioning"),
            )
        ax.set_yscale("log")
        ax.set_xticks(x, ["Field", "Bound"])
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", which="both", alpha=0.18)
    axes[0].set_ylabel("Median neighborhood JS change (bits)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Neighborhood change concentrates in bound regions and is largely kinematic", y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "camels_phase_control.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "camels_phase_control.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    style()
    model_comparison()
    neighborhood_control()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
