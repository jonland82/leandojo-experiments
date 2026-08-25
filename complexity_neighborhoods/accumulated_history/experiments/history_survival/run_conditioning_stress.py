"""Stress-test compatible-history survival under imperfect conditioning.

The exact-conditioning validation establishes the estimator's ideal behavior.
This stage measures how parameter error and coarse state resolution leak
predictable dynamics into an apparent conditional-information rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from run_history_survival import SEED, standard_map, wrapped_delta


TARGET_BITS_PER_GYR = 0.25
HUBBLE_RATE_PER_GYR = 1.0 / 13.8
HUBBLE_DT_GYR = 0.25
HUBBLE_STEPS = 40
HUBBLE_PAIRS = 500_000
HUBBLE_FRACTIONAL_ERRORS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
HUBBLE_RADII = (0.0025, 0.005, 0.01)

CHAOS_PAIRS = 150_000
CHAOS_STEPS = 30
TRUE_MAP_STRENGTH = 2.0
MAP_FRACTIONAL_ERRORS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
QUANTIZATION_WIDTHS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)
CHAOS_RADII = (0.05, 0.1, 0.2)


@dataclass(frozen=True)
class Leakage:
    setting: float
    radius: float
    survival: float
    complexity: float
    rate: float
    rate_standard_error: float


def leakage_from_distances(
    distances: np.ndarray, radius: float, duration: float, setting: float
) -> Leakage:
    """Measure fixed-radius leakage and its binomial sampling uncertainty."""
    survivors = int(np.count_nonzero(distances <= radius))
    count = len(distances)
    survival = survivors / count
    if survivors == 0:
        return Leakage(setting, radius, 0.0, math.inf, math.inf, math.inf)
    complexity = 0.0 if survivors == count else -math.log2(survival)
    if survivors == count:
        standard_error = 0.0
    else:
        standard_error = math.sqrt((1.0 - survival) / (count * survival)) / math.log(2.0)
    return Leakage(
        setting,
        radius,
        survival,
        complexity,
        complexity / duration,
        standard_error / duration,
    )


def hubble_stress() -> list[Leakage]:
    """Leakage caused by conditioning on a slightly incorrect Hubble rate."""
    rng = np.random.default_rng(SEED + 401)
    first = rng.normal(0.0, 1.0, size=(HUBBLE_PAIRS, 3))
    second = rng.normal(0.0, 1.0, size=(HUBBLE_PAIRS, 3))
    initial_separation = np.linalg.norm(first - second, axis=1)
    rms_separation = math.sqrt(float(np.mean(initial_separation**2)))

    true_growth = math.exp(HUBBLE_RATE_PER_GYR * HUBBLE_DT_GYR)
    geometric_sum = sum(true_growth**step for step in range(HUBBLE_STEPS))
    duration = HUBBLE_STEPS * HUBBLE_DT_GYR
    rows: list[Leakage] = []
    for fractional_error in HUBBLE_FRACTIONAL_ERRORS:
        model_rate = HUBBLE_RATE_PER_GYR * (1.0 + fractional_error)
        model_growth = math.exp(model_rate * HUBBLE_DT_GYR)
        normalized_distance = (
            initial_separation
            * abs(true_growth - model_growth)
            * geometric_sum
            / rms_separation
        )
        for radius in HUBBLE_RADII:
            rows.append(
                leakage_from_distances(
                    normalized_distance, radius, duration, fractional_error
                )
            )
    return rows


def chaos_parameter_stress() -> list[Leakage]:
    """Leakage caused by a misspecified deterministic force parameter."""
    rng = np.random.default_rng(SEED + 402)
    base_first = rng.uniform(0.0, 2.0 * math.pi, size=(CHAOS_PAIRS, 2))
    base_second = (
        base_first + rng.normal(0.0, 1.0e-8, size=(CHAOS_PAIRS, 2))
    ) % (2.0 * math.pi)
    rows: list[Leakage] = []

    for fractional_error in MAP_FRACTIONAL_ERRORS:
        first = base_first.copy()
        second = base_second.copy()
        model_strength = TRUE_MAP_STRENGTH * (1.0 + fractional_error)
        cumulative = np.zeros(CHAOS_PAIRS)
        for _ in range(CHAOS_STEPS):
            next_first = standard_map(first, TRUE_MAP_STRENGTH)
            next_second = standard_map(second, TRUE_MAP_STRENGTH)
            predicted_first = standard_map(first, model_strength)
            predicted_second = standard_map(second, model_strength)
            innovation_first = wrapped_delta(next_first, predicted_first)
            innovation_second = wrapped_delta(next_second, predicted_second)
            cumulative += np.linalg.norm(
                innovation_first - innovation_second, axis=1
            )
            first, second = next_first, next_second
        for radius in CHAOS_RADII:
            rows.append(
                leakage_from_distances(
                    cumulative, radius, float(CHAOS_STEPS), fractional_error
                )
            )
    return rows


def quantize_state(state: np.ndarray, width: float) -> np.ndarray:
    if width == 0.0:
        return state
    return (np.round(state / width) * width) % (2.0 * math.pi)


def chaos_resolution_stress() -> list[Leakage]:
    """Leakage when the correct dynamics receive a coarse current state."""
    rng = np.random.default_rng(SEED + 403)
    base_first = rng.uniform(0.0, 2.0 * math.pi, size=(CHAOS_PAIRS, 2))
    base_second = (
        base_first + rng.normal(0.0, 1.0e-8, size=(CHAOS_PAIRS, 2))
    ) % (2.0 * math.pi)
    rows: list[Leakage] = []

    for width in QUANTIZATION_WIDTHS:
        first = base_first.copy()
        second = base_second.copy()
        cumulative = np.zeros(CHAOS_PAIRS)
        for _ in range(CHAOS_STEPS):
            next_first = standard_map(first, TRUE_MAP_STRENGTH)
            next_second = standard_map(second, TRUE_MAP_STRENGTH)
            predicted_first = standard_map(
                quantize_state(first, width), TRUE_MAP_STRENGTH
            )
            predicted_second = standard_map(
                quantize_state(second, width), TRUE_MAP_STRENGTH
            )
            innovation_first = wrapped_delta(next_first, predicted_first)
            innovation_second = wrapped_delta(next_second, predicted_second)
            cumulative += np.linalg.norm(
                innovation_first - innovation_second, axis=1
            )
            first, second = next_first, next_second
        for radius in CHAOS_RADII:
            rows.append(
                leakage_from_distances(cumulative, radius, float(CHAOS_STEPS), width)
            )
    return rows


def select_radius(rows: list[Leakage], radius: float) -> list[Leakage]:
    return [row for row in rows if math.isclose(row.radius, radius)]


def monotonic_non_decreasing(values: list[float], tolerance: float = 1.0e-12) -> bool:
    return all(second + tolerance >= first for first, second in zip(values, values[1:]))


def format_rate(row: Leakage, unit: str) -> str:
    if not math.isfinite(row.rate):
        return f"infinite {unit} (no survivors)"
    return f"{row.rate:.4f} +/- {row.rate_standard_error:.4f} {unit}"


def main() -> None:
    print("Compatible-history survival: imperfect-conditioning stress test")
    print(f"Seed={SEED}")
    print()

    hubble_rows = hubble_stress()
    print("1. Incorrect Hubble-rate conditioning")
    print(
        f"   true H={HUBBLE_RATE_PER_GYR:.5f}/Gyr, duration="
        f"{HUBBLE_STEPS * HUBBLE_DT_GYR:.1f} Gyr"
    )
    print("   central fixed radius r=0.005 (normalized cumulative residual):")
    for row in select_radius(hubble_rows, 0.005):
        print(
            f"     H error={100.0 * row.setting:>4.1f}%: "
            f"rho={row.survival:.6f}, {format_rate(row, 'bits/Gyr')}"
        )
    print("   target-crossing sensitivity by fixed radius:")
    for radius in HUBBLE_RADII:
        candidates = [
            row
            for row in select_radius(hubble_rows, radius)
            if row.rate >= TARGET_BITS_PER_GYR
        ]
        crossing = candidates[0].setting if candidates else math.nan
        if math.isnan(crossing):
            message = "not reached"
        else:
            message = f"{100.0 * crossing:.1f}% H error"
        print(f"     r={radius:.4f}: {message}")
    print()

    parameter_rows = chaos_parameter_stress()
    print("2. Incorrect deterministic force parameter")
    print("   central fixed radius r=0.1:")
    for row in select_radius(parameter_rows, 0.1):
        print(
            f"     force error={100.0 * row.setting:>4.1f}%: "
            f"rho={row.survival:.6f}, {format_rate(row, 'bits/update')}"
        )
    print()

    resolution_rows = chaos_resolution_stress()
    print("3. Coarse-state conditioning with the correct force law")
    print("   central fixed radius r=0.1:")
    for row in select_radius(resolution_rows, 0.1):
        print(
            f"     state bin={row.setting:>7.4f} rad: "
            f"rho={row.survival:.6f}, {format_rate(row, 'bits/update')}"
        )
    print()

    hubble_central = select_radius(hubble_rows, 0.005)
    parameter_central = select_radius(parameter_rows, 0.1)
    resolution_central = select_radius(resolution_rows, 0.1)
    checks = {
        "exact Hubble model has zero leakage": hubble_central[0].rate == 0.0,
        "exact force model has zero leakage": parameter_central[0].rate == 0.0,
        "exact state has zero leakage": resolution_central[0].rate == 0.0,
        "Hubble leakage grows with error": monotonic_non_decreasing(
            [row.rate for row in hubble_central]
        ),
        "force leakage grows with error": monotonic_non_decreasing(
            [row.rate for row in parameter_central]
        ),
        "resolution leakage grows with coarseness": monotonic_non_decreasing(
            [row.rate for row in resolution_central]
        ),
    }
    print("Validation checks")
    for name, passed in checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
