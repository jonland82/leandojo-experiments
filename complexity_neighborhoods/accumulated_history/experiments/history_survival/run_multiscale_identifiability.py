"""Test whether compatible-history rates are identifiable across tolerance.

The generator applies uniform physical expansion, but the conditioning model is
the H-free identity propagator. Compatibility radii are expressed only in units
of the RMS shared-past uncertainty. The experiment asks whether the inferred
rate has a stable one-decade plateau and transfers across expansion histories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from run_history_survival import SEED


PAIR_COUNT = 1_000_000
DURATION_GYR = 10.0
REFERENCE_H_PER_GYR = 1.0 / 13.8
H_FACTORS = (0.0, 0.5, 1.0, 1.5)
RADII = np.logspace(math.log10(0.01), math.log10(5.0), 151)
TARGET_RATE = 0.25
PLATEAU_MIN_RATE = 0.025
PLATEAU_MAX_RELATIVE_SPAN = 0.10
PLATEAU_RADIUS_RATIO = 10.0
SURVIVAL_FLOOR = 1.0e-4
SURVIVAL_CEILING = 0.99


@dataclass(frozen=True)
class SurvivalCurve:
    h_factor: float
    distances: np.ndarray
    survival: np.ndarray
    rate: np.ndarray


def normalized_initial_separation(pair_count: int = PAIR_COUNT) -> np.ndarray:
    """Return pair separations in independently fixed shared-past RMS units."""
    rng = np.random.default_rng(SEED + 501)
    first = rng.normal(0.0, 1.0, size=(pair_count, 3))
    second = rng.normal(0.0, 1.0, size=(pair_count, 3))
    separation = np.linalg.norm(first - second, axis=1)
    return separation / math.sqrt(float(np.mean(separation**2)))


def h_free_residual_distance(
    normalized_separation: np.ndarray, h_factor: float
) -> np.ndarray:
    """Cumulative residual against an identity predictor containing no H.

    Uniform expansion is monotonic, so the sum of stepwise pair-residual norms
    telescopes to the total fractional growth times the initial separation.
    """
    total_growth = math.exp(
        h_factor * REFERENCE_H_PER_GYR * DURATION_GYR
    )
    return abs(total_growth - 1.0) * normalized_separation


def survival_curve(
    normalized_separation: np.ndarray, h_factor: float
) -> SurvivalCurve:
    distances = h_free_residual_distance(normalized_separation, h_factor)
    ordered = np.sort(distances)
    counts = np.searchsorted(ordered, RADII, side="right")
    survival = counts / len(ordered)
    rate = np.full_like(survival, math.inf, dtype=float)
    positive = survival > 0.0
    rate[positive] = -np.log2(survival[positive]) / DURATION_GYR
    rate[survival == 1.0] = 0.0
    return SurvivalCurve(h_factor, distances, survival, rate)


def find_plateau(curve: SurvivalCurve) -> tuple[float, float] | None:
    """Find a nontrivial one-decade interval stable to 10% in inferred rate."""
    for start, lower in enumerate(RADII):
        end = int(np.searchsorted(RADII, lower * PLATEAU_RADIUS_RATIO, side="left"))
        if end >= len(RADII):
            break
        interval_rate = curve.rate[start : end + 1]
        interval_survival = curve.survival[start : end + 1]
        valid = (
            np.all(np.isfinite(interval_rate))
            and np.all(interval_rate >= PLATEAU_MIN_RATE)
            and np.all(interval_survival >= SURVIVAL_FLOOR)
            and np.all(interval_survival <= SURVIVAL_CEILING)
        )
        if not valid:
            continue
        median = float(np.median(interval_rate))
        relative_span = float(np.ptp(interval_rate) / median)
        if relative_span <= PLATEAU_MAX_RELATIVE_SPAN:
            return float(lower), float(RADII[end])
    return None


def closest_to_target(curve: SurvivalCurve) -> tuple[float, float, float]:
    finite = np.isfinite(curve.rate)
    indices = np.flatnonzero(finite)
    index = indices[int(np.argmin(np.abs(curve.rate[finite] - TARGET_RATE)))]
    return float(RADII[index]), float(curve.rate[index]), float(curve.survival[index])


def rate_at_radius(curve: SurvivalCurve, radius: float) -> float:
    index = int(np.argmin(np.abs(RADII - radius)))
    return float(curve.rate[index])


def resolution_collapse_check() -> float:
    """Check normalization across three independently declared length units."""
    rng = np.random.default_rng(SEED + 502)
    first = rng.normal(0.0, 1.0, size=(200_000, 3))
    second = rng.normal(0.0, 1.0, size=(200_000, 3))
    curves = []
    for scale in (0.5, 1.0, 2.0):
        separation = np.linalg.norm(scale * first - scale * second, axis=1)
        normalized = separation / math.sqrt(float(np.mean(separation**2)))
        curves.append(survival_curve(normalized, 1.0).rate)
    maximum = 0.0
    for first_curve in curves:
        for second_curve in curves:
            if not np.array_equal(np.isinf(first_curve), np.isinf(second_curve)):
                return math.inf
            finite = np.isfinite(first_curve) & np.isfinite(second_curve)
            if np.any(finite):
                maximum = max(
                    maximum,
                    float(np.max(np.abs(first_curve[finite] - second_curve[finite]))),
                )
    return maximum


def timestep_invariance_check() -> float:
    """Check that cumulative monotonic expansion is independent of slicing."""
    rates = []
    for steps in (20, 40, 80):
        dt = DURATION_GYR / steps
        growth = math.exp(REFERENCE_H_PER_GYR * dt)
        cumulative_growth = sum(
            abs(growth - 1.0) * growth**step for step in range(steps)
        )
        rates.append(cumulative_growth)
    return float(max(rates) - min(rates))


def main() -> None:
    print("Compatible-history survival: multiscale identifiability gate")
    print(f"Seed={SEED}; pairs={PAIR_COUNT:,}; duration={DURATION_GYR:.1f} Gyr")
    print("Conditioner: identity propagator (contains no H)")
    print("Tolerance units: RMS initial shared-past separation")
    print()

    normalized = normalized_initial_separation()
    curves = [survival_curve(normalized, factor) for factor in H_FACTORS]

    print("Full-radius scan")
    target_choices: dict[float, tuple[float, float, float]] = {}
    plateaus: dict[float, tuple[float, float] | None] = {}
    for curve in curves:
        plateau = find_plateau(curve)
        plateaus[curve.h_factor] = plateau
        radius, rate, survival = closest_to_target(curve)
        target_choices[curve.h_factor] = (radius, rate, survival)
        plateau_text = "none" if plateau is None else f"{plateau[0]:.4f}--{plateau[1]:.4f}"
        print(
            f"   H/H_ref={curve.h_factor:>3.1f}: plateau={plateau_text}; "
            f"closest to 0.25 bits/Gyr at r={radius:.4f} "
            f"(rate={rate:.4f}, rho={survival:.4f})"
        )
    print()

    calibration_radius = target_choices[1.0][0]
    print(
        "One-radius transfer after calibrating the reference history to "
        f"0.25 bits/Gyr (r={calibration_radius:.4f})"
    )
    transfer_errors = []
    for curve in curves:
        measured = rate_at_radius(curve, calibration_radius)
        expected = TARGET_RATE * curve.h_factor
        error = abs(measured - expected)
        if curve.h_factor > 0.0:
            transfer_errors.append(error / expected)
        print(
            f"   H/H_ref={curve.h_factor:>3.1f}: measured={measured:.4f}, "
            f"linear target={expected:.4f} bits/Gyr"
        )
    print()

    resolution_spread = resolution_collapse_check()
    timestep_spread = timestep_invariance_check()
    print("Nuisance checks")
    print(f"   normalized-resolution maximum rate spread: {resolution_spread:.3e} bits/Gyr")
    print(f"   timestep cumulative-distance spread:        {timestep_spread:.3e}")
    print()

    checks = {
        "static history gives zero rate": bool(np.all(curves[0].rate == 0.0)),
        "initial-resolution normalization collapses": resolution_spread < 1.0e-12,
        "time slicing is invariant": timestep_spread < 1.0e-12,
        "reference history has a one-decade plateau": plateaus[1.0] is not None,
        "one calibration transfers within 10%": max(transfer_errors) < 0.10,
    }
    print("Identifiability checks")
    for name, passed in checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
