"""Generate the two vector figures used by the compatible-histories note."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments" / "history_survival"
sys.path.insert(0, str(EXPERIMENTS))

from run_multiscale_identifiability import (  # noqa: E402
    RADII,
    closest_to_target,
    normalized_initial_separation,
    survival_curve,
)
from run_noether_invariance import (  # noqa: E402
    ANHARMONIC_LAMBDA,
    ASPECTS,
    BAND_RESOLUTIONS,
    CELL_RESOLUTIONS,
    band_density_results,
    canonical_shear_count,
    exact_band_density,
    rectangular_harmonic_count,
)


INK = "#1F2933"
MUTED = "#66737F"
ACCENT = "#394B59"
PALE = "#AAB4BC"
LIGHT = "#D6DDE2"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def radius_figure() -> None:
    normalized = normalized_initial_separation()
    factors = (0.5, 1.0, 1.5)
    colors = (PALE, ACCENT, INK)
    linestyles = ("--", "-", "-.")

    fig, ax = plt.subplots(figsize=(6.7, 3.05))
    for factor, color, linestyle in zip(factors, colors, linestyles):
        curve = survival_curve(normalized, factor)
        finite = np.isfinite(curve.rate)
        ax.plot(
            RADII[finite],
            curve.rate[finite],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=rf"${factor:.1f}H_{{\rm ref}}$",
        )
        radius, rate, _ = closest_to_target(curve)
        ax.scatter(
            radius,
            rate,
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.axhline(0.25, color=MUTED, linestyle=(0, (3, 3)), linewidth=1.0)
    ax.text(0.0108, 0.263, "same apparent target", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 5.0)
    ax.set_ylim(0.0, 0.72)
    ax.set_xlabel(r"Compatibility radius $r$ (initial RMS-separation units)")
    ax.set_ylabel(r"Inferred rate (bits/Gyr)")
    ax.legend(loc="upper right", ncol=3, columnspacing=1.2, handlelength=2.2)
    ax.grid(axis="y", color=LIGHT, linewidth=0.55, alpha=0.75)
    fig.tight_layout(pad=0.5)
    fig.savefig(HERE / "radius_identifiability.pdf", bbox_inches="tight")
    plt.close(fig)


def invariance_figure() -> None:
    scheme_specs = (
        ("aspect 0.5", lambda eps: rectangular_harmonic_count(eps, ASPECTS[0]), PALE, "o"),
        ("aspect 1", lambda eps: rectangular_harmonic_count(eps, ASPECTS[1]), ACCENT, "s"),
        ("aspect 2", lambda eps: rectangular_harmonic_count(eps, ASPECTS[2]), "#87949E", "^"),
        ("canonical shear", canonical_shear_count, INK, "D"),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.7, 3.05))
    refinement_index = np.arange(len(CELL_RESOLUTIONS))
    for label, calculator, color, marker in scheme_specs:
        values = [calculator(eps).finite_bits for eps in CELL_RESOLUTIONS]
        left.plot(
            refinement_index,
            values,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.35,
            label=label,
        )
    left.set_xticks(refinement_index)
    left.set_xticklabels(("0.1", "0.05", "0.025", "0.0125"))
    left.set_xlabel(r"Cell scale $\epsilon$ (finer $\longrightarrow$)")
    left.set_ylabel("Finite cell-count term (bits)")
    left.set_title("Cell incidence retains the grid")
    left.grid(axis="y", color=LIGHT, linewidth=0.55, alpha=0.75)

    rows = band_density_results()
    finest = BAND_RESOLUTIONS[-1]
    labels = [spec[0] for spec in scheme_specs]
    x = np.arange(len(labels))
    width = 0.34
    for offset, (name, quartic_lambda, color, hatch) in enumerate(
        (
            ("harmonic", 0.0, ACCENT, None),
            ("anharmonic", ANHARMONIC_LAMBDA, PALE, "///"),
        )
    ):
        exact = exact_band_density(quartic_lambda)
        selected = [
            row for row in rows if row.hamiltonian == name and row.epsilon == finest
        ]
        errors = [100.0 * (row.density - exact) / exact for row in selected]
        right.bar(
            x + (offset - 0.5) * width,
            errors,
            width,
            color=color,
            hatch=hatch,
            edgecolor=INK if hatch else color,
            linewidth=0.5,
            label=name,
        )
    right.axhline(0.0, color=MUTED, linewidth=0.8)
    right.set_xticks(
        x,
        ("aspect\n0.5", "aspect\n1", "aspect\n2", "canonical\nshear"),
    )
    right.set_ylabel("Liouville-density error (%)")
    right.set_title("Phase volume forgets the grid")
    right.legend(loc="upper left")
    right.grid(axis="y", color=LIGHT, linewidth=0.55, alpha=0.75)

    handles, legend_labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.30, -0.01),
        columnspacing=0.9,
        handlelength=1.5,
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.25, wspace=0.36)
    fig.savefig(HERE / "measure_invariance.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    radius_figure()
    invariance_figure()


if __name__ == "__main__":
    main()
