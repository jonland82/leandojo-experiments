"""Generate the vector figure for the invariant-history paper."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments" / "history_survival"
sys.path.insert(0, str(EXPERIMENTS))

from run_constrained_history_gauge import (  # noqa: E402
    SOBOL_POWERS,
    nonlinear_gauge_estimate,
    reduced_estimate,
)
from run_history_measure_extensions import (  # noqa: E402
    ANHARMONIC_STEPS,
    anharmonic_statistics,
    initial_ensemble,
)


INK = "#1F2933"
MUTED = "#66737F"
ACCENT = "#394B59"
PALE = "#AAB4BC"
LIGHT = "#D6DDE2"
TEXT = "#000000"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def main() -> None:
    set_style()
    x0, p0 = initial_ensemble()
    discrepancies = []
    for dt in ANHARMONIC_STEPS:
        physical = anharmonic_statistics(x0, p0, dt, transformed=False)
        transformed = anharmonic_statistics(x0, p0, dt, transformed=True)
        discrepancies.append(abs(physical[0] - transformed[0]))

    power = SOBOL_POWERS[-1]
    reference = reduced_estimate(power)
    gauges = ("t+alpha*x*p=0", "t+beta*x^2/2=0")
    gauge_rows = [nonlinear_gauge_estimate(gauge, power) for gauge in gauges]
    corrected = [
        abs(row.survival_bits - reference.survival_bits) for row in gauge_rows
    ]
    naive = [
        abs(row.naive_survival_bits - reference.survival_bits)
        for row in gauge_rows
    ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.15))
    left.loglog(
        ANHARMONIC_STEPS,
        discrepancies,
        color=ACCENT,
        marker="o",
        linewidth=1.6,
        label="measured discrepancy",
    )
    reference_curve = discrepancies[-1] * (
        np.asarray(ANHARMONIC_STEPS) / ANHARMONIC_STEPS[-1]
    ) ** 2
    left.loglog(
        ANHARMONIC_STEPS,
        reference_curve,
        color=PALE,
        linestyle="--",
        linewidth=1.2,
        label=r"second-order reference",
    )
    left.invert_xaxis()
    left.set_xticks(ANHARMONIC_STEPS)
    left.set_xticklabels(("0.10", "0.05", "0.025", "0.0125"))
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set_xlabel(r"Time step $\Delta t$ (finer $\longrightarrow$)")
    left.set_ylabel(r"$|C_{Q,P}-C_{x,p}|$ (bits)")
    left.set_ylim(1.4e-5, 2.6e-3)
    left.set_title("Canonical discrepancy vanishes", fontsize=10, pad=8)
    left.set_axisbelow(True)
    left.grid(which="major", axis="y", color=LIGHT, linewidth=0.65, alpha=0.8)
    left.legend(loc="lower left", fontsize=7.4, handlelength=2.2)

    positions = np.arange(2)
    width = 0.34
    right.bar(
        positions - width / 2,
        corrected,
        width,
        color=ACCENT,
        label="with determinant",
    )
    right.bar(
        positions + width / 2,
        naive,
        width,
        color=PALE,
        edgecolor=INK,
        linewidth=0.45,
        hatch="///",
        label="determinant omitted",
    )
    right.set_yscale("log")
    right.set_ylim(3.0e-6, 8.0e-1)
    right.set_xticks(positions, (r"$t+0.15xp$", r"$t+0.10x^2$"))
    right.set_ylabel(r"$|C_\chi-C_{t=0}|$ (bits)")
    right.set_title("Gauge determinant removes false bits", fontsize=10, pad=8)
    right.set_axisbelow(True)
    right.grid(which="major", axis="y", color=LIGHT, linewidth=0.65, alpha=0.8)
    right.legend(loc="upper right", fontsize=7.4, handlelength=2.2)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.19, wspace=0.42)
    fig.savefig(HERE / "history_measure_invariance.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
