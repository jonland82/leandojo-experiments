"""Gauge test for an invariant measure on complete classical histories.

The parametrized harmonic oscillator has extended phase-space constraint

    C = p_t + (p**2 + x**2)/2 = 0.

Its physical histories are labelled by the Dirac observables (X, P), the
position and momentum at physical time zero.  We compare direct integration
over (X, P) with integrations on nonlinear gauge surfaces.  The latter must
include the Faddeev--Popov factor |{chi, C}|.  Omitting it supplies a controlled
failure case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


BOUND = 1.5
SURFACE_BOUND = math.sqrt(2.0) * BOUND
DURATION = 4.0
RECORD_SCALE = 0.35
SOBOL_POWERS = (16, 18, 20)
GAUGE_XP_ALPHA = 0.15
GAUGE_X2_BETA = 0.20


@dataclass(frozen=True)
class Estimate:
    gauge: str
    power: int
    area: float
    survival_bits: float
    kl_bits: float
    naive_survival_bits: float


def history_log_weight(initial_x: np.ndarray, initial_p: np.ndarray) -> np.ndarray:
    """Exact continuous-record compatibility for a complete harmonic history."""
    delta_x = initial_x - 1.0
    cosine_squared = 0.5 * DURATION + 0.25 * math.sin(2.0 * DURATION)
    sine_squared = 0.5 * DURATION - 0.25 * math.sin(2.0 * DURATION)
    cosine_sine = 0.5 * math.sin(DURATION) ** 2
    integrated_residual = (
        cosine_squared * delta_x**2
        + sine_squared * initial_p**2
        + 2.0 * cosine_sine * delta_x * initial_p
    )
    return -0.5 * integrated_residual / RECORD_SCALE**2


def weighted_statistics(
    log_weight: np.ndarray, prior_weight: np.ndarray
) -> tuple[float, float]:
    positive = prior_weight > 0.0
    log_weight = log_weight[positive]
    prior_weight = prior_weight[positive]
    maximum = float(np.max(log_weight))
    compatibility = np.exp(log_weight - maximum)
    prior_total = float(np.sum(prior_weight))
    posterior_unnormalized = prior_weight * compatibility
    posterior_total = float(np.sum(posterior_unnormalized))
    log_z = maximum + math.log(posterior_total / prior_total)
    posterior_mean_log_weight = float(
        np.sum(posterior_unnormalized * log_weight) / posterior_total
    )
    survival_bits = -log_z / math.log(2.0)
    kl_bits = (posterior_mean_log_weight - log_z) / math.log(2.0)
    return survival_bits, kl_bits


def dirac_observables(
    x: np.ndarray, p: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(time)
    sine = np.sin(time)
    initial_x = x * cosine - p * sine
    initial_p = x * sine + p * cosine
    return initial_x, initial_p


def reduced_estimate(power: int) -> Estimate:
    unit = qmc.Sobol(d=2, scramble=True, seed=20260826).random_base2(power)
    initial = -BOUND + 2.0 * BOUND * unit
    log_weight = history_log_weight(initial[:, 0], initial[:, 1])
    prior = np.ones(log_weight.shape)
    survival, kl = weighted_statistics(log_weight, prior)
    return Estimate("t=0", power, (2.0 * BOUND) ** 2, survival, kl, survival)


def nonlinear_gauge_estimate(gauge: str, power: int) -> Estimate:
    seed = 20260827 if gauge == "t+alpha*x*p=0" else 20260828
    unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(power)
    surface = -SURFACE_BOUND + 2.0 * SURFACE_BOUND * unit
    x = surface[:, 0]
    p = surface[:, 1]

    if gauge == "t+alpha*x*p=0":
        time = -GAUGE_XP_ALPHA * x * p
        fp = 1.0 + GAUGE_XP_ALPHA * (p**2 - x**2)
    elif gauge == "t+beta*x^2/2=0":
        time = -0.5 * GAUGE_X2_BETA * x**2
        fp = 1.0 + GAUGE_X2_BETA * x * p
    else:
        raise ValueError(gauge)

    if np.any(fp <= 0.0):
        raise RuntimeError(f"Gauge {gauge} has a zero or negative FP determinant")

    initial_x, initial_p = dirac_observables(x, p, time)
    physical_domain = (
        (np.abs(initial_x) <= BOUND) & (np.abs(initial_p) <= BOUND)
    )
    log_weight = history_log_weight(initial_x, initial_p)
    correct_prior = physical_domain.astype(float) * fp
    naive_prior = physical_domain.astype(float)
    survival, kl = weighted_statistics(log_weight, correct_prior)
    naive_survival, _ = weighted_statistics(log_weight, naive_prior)
    sample_area = (2.0 * SURFACE_BOUND) ** 2
    quotient_area = sample_area * float(np.mean(correct_prior))
    return Estimate(
        gauge,
        power,
        quotient_area,
        survival,
        kl,
        naive_survival,
    )


def main() -> None:
    gauges = ("t+alpha*x*p=0", "t+beta*x^2/2=0")
    all_rows: list[Estimate] = []
    print("Parametrized-oscillator gauge test")
    print("constraint: C = p_t + (p^2+x^2)/2 = 0")
    print(f"target reduced Liouville area: {(2.0 * BOUND) ** 2:.6f}")
    print()
    print("samples      gauge                 area       C_FP bits   KL_FP bits   C_naive")
    for power in SOBOL_POWERS:
        rows = [reduced_estimate(power)] + [
            nonlinear_gauge_estimate(gauge, power) for gauge in gauges
        ]
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{2**power:7,d}   {row.gauge:21s}  {row.area:9.6f}   "
                f"{row.survival_bits:9.6f}   {row.kl_bits:9.6f}   "
                f"{row.naive_survival_bits:9.6f}"
            )
        print()

    finest = [row for row in all_rows if row.power == SOBOL_POWERS[-1]]
    reference = next(row for row in finest if row.gauge == "t=0")
    nonlinear = [row for row in finest if row.gauge != "t=0"]
    fp_c_errors = [abs(row.survival_bits - reference.survival_bits) for row in nonlinear]
    fp_kl_errors = [abs(row.kl_bits - reference.kl_bits) for row in nonlinear]
    area_errors = [abs(row.area - reference.area) / reference.area for row in nonlinear]
    naive_errors = [
        abs(row.naive_survival_bits - reference.survival_bits) for row in nonlinear
    ]

    previous = [row for row in all_rows if row.power == SOBOL_POWERS[-2]]
    previous_by_gauge = {row.gauge: row for row in previous}
    refinement_changes = [
        abs(row.survival_bits - previous_by_gauge[row.gauge].survival_bits)
        for row in finest
    ]

    checks = {
        "gauge-fixed quotient area": max(area_errors) < 5.0e-4,
        "FP-weighted survival information is gauge invariant": max(fp_c_errors) < 5.0e-4,
        "FP-weighted KL information is gauge invariant": max(fp_kl_errors) < 5.0e-4,
        "gauge estimates converge with ensemble refinement": max(refinement_changes) < 5.0e-4,
        "omitting FP determinant produces a gauge artifact": min(naive_errors) > 0.005,
    }
    print("Checks")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"maximum FP-weighted C discrepancy: {max(fp_c_errors):.6f} bits")
    print(f"maximum FP-weighted KL discrepancy: {max(fp_kl_errors):.6f} bits")
    print(f"maximum quotient-area relative error: {100.0 * max(area_errors):.5f}%")
    print(f"smallest naive gauge artifact: {min(naive_errors):.6f} bits")
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
