"""Invariance gate for Noether-shell cardinality and its finite part.

The experiment compares raw intersecting-cell counts under grid anisotropy and
a canonical shear, then replaces cell incidence by Liouville band volume. It
also repeats the measure calculation for an anharmonic Hamiltonian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


BOUND = 2.0
ENERGY = 1.0
CELL_RESOLUTIONS = (0.1, 0.05, 0.025, 0.0125)
BAND_RESOLUTIONS = (0.04, 0.02, 0.01, 0.005)
ASPECTS = (0.5, 1.0, 2.0)
SHEAR = 1.0
ENERGY_BAND = 0.05
ANHARMONIC_LAMBDA = 0.5


@dataclass(frozen=True)
class CellCount:
    scheme: str
    epsilon: float
    admissible: int
    total: int
    bits: float
    finite_bits: float


@dataclass(frozen=True)
class BandDensity:
    hamiltonian: str
    scheme: str
    epsilon: float
    density: float


def centers_for_width(lower: float, upper: float, width: float) -> np.ndarray:
    count = int(round((upper - lower) / width))
    if not math.isclose(count * width, upper - lower, abs_tol=1.0e-12):
        raise ValueError("Cell width must tile its coordinate range")
    return lower + 0.5 * width + width * np.arange(count)


def squared_extrema(centers: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    lower = centers - 0.5 * width
    upper = centers + 0.5 * width
    minimum_absolute = np.where(
        (lower <= 0.0) & (upper >= 0.0),
        0.0,
        np.minimum(np.abs(lower), np.abs(upper)),
    )
    maximum_absolute = np.maximum(np.abs(lower), np.abs(upper))
    return minimum_absolute**2, maximum_absolute**2


def rectangular_harmonic_count(epsilon: float, aspect: float) -> CellCount:
    dx = epsilon * aspect
    dp = epsilon / aspect
    x = centers_for_width(-BOUND, BOUND, dx)
    p = centers_for_width(-BOUND, BOUND, dp)
    x_min, x_max = squared_extrema(x, dx)
    p_min, p_max = squared_extrema(p, dp)
    h_min = 0.5 * (x_min[:, None] + p_min[None, :])
    h_max = 0.5 * (x_max[:, None] + p_max[None, :])
    admissible = int(np.count_nonzero((h_min <= ENERGY) & (h_max >= ENERGY)))
    total = h_min.size
    bits = -math.log2(admissible / total)
    return CellCount(
        scheme=f"aspect={aspect:g}",
        epsilon=epsilon,
        admissible=admissible,
        total=total,
        bits=bits,
        finite_bits=bits - math.log2(1.0 / epsilon),
    )


def sheared_harmonic_extrema(
    q: np.ndarray, p: np.ndarray, width: float, shear: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact extrema of H=[q^2+(P-s*q)^2]/2 over rectangular cells."""
    q_low = q[:, None] - 0.5 * width
    q_high = q[:, None] + 0.5 * width
    p_low = p[None, :] - 0.5 * width
    p_high = p[None, :] + 0.5 * width

    def energy(q_value: np.ndarray, p_value: np.ndarray) -> np.ndarray:
        return 0.5 * (q_value**2 + (p_value - shear * q_value) ** 2)

    maximum = np.maximum.reduce(
        (
            energy(q_low, p_low),
            energy(q_low, p_high),
            energy(q_high, p_low),
            energy(q_high, p_high),
        )
    )

    candidates = []
    for q_edge in (q_low, q_high):
        p_optimum = np.clip(shear * q_edge, p_low, p_high)
        candidates.append(energy(q_edge, p_optimum))
    denominator = 1.0 + shear**2
    for p_edge in (p_low, p_high):
        q_optimum = np.clip(shear * p_edge / denominator, q_low, q_high)
        candidates.append(energy(q_optimum, p_edge))
    minimum = np.minimum.reduce(candidates)
    interior = (
        (q_low <= 0.0)
        & (q_high >= 0.0)
        & (p_low <= 0.0)
        & (p_high >= 0.0)
    )
    minimum = np.where(interior, 0.0, minimum)
    return minimum, maximum


def canonical_shear_count(epsilon: float) -> CellCount:
    q = centers_for_width(-BOUND, BOUND, epsilon)
    p_extent = BOUND * (1.0 + abs(SHEAR))
    p = centers_for_width(-p_extent, p_extent, epsilon)
    h_min, h_max = sheared_harmonic_extrema(q, p, epsilon, SHEAR)
    q_center = q[:, None]
    p_center = p[None, :]
    physical_domain = (np.abs(q_center) <= BOUND) & (
        np.abs(p_center - SHEAR * q_center) <= BOUND
    )
    admissible_mask = physical_domain & (h_min <= ENERGY) & (h_max >= ENERGY)
    admissible = int(np.count_nonzero(admissible_mask))
    total = int(np.count_nonzero(physical_domain))
    bits = -math.log2(admissible / total)
    return CellCount(
        scheme="canonical shear=1",
        epsilon=epsilon,
        admissible=admissible,
        total=total,
        bits=bits,
        finite_bits=bits - math.log2(1.0 / epsilon),
    )


def potential(x: np.ndarray | float, quartic_lambda: float) -> np.ndarray | float:
    return 0.5 * np.asarray(x) ** 2 + 0.25 * quartic_lambda * np.asarray(x) ** 4


def turning_point(energy: float, quartic_lambda: float) -> float:
    if quartic_lambda == 0.0:
        return math.sqrt(2.0 * energy)
    squared = (-1.0 + math.sqrt(1.0 + 4.0 * quartic_lambda * energy)) / quartic_lambda
    return math.sqrt(squared)


def phase_volume(energy: float, quartic_lambda: float) -> float:
    """Liouville volume enclosed by p^2/2+V(x)<=energy."""
    x_turn = turning_point(energy, quartic_lambda)

    def momentum_width(x: float) -> float:
        available = max(0.0, energy - float(potential(x, quartic_lambda)))
        return 2.0 * math.sqrt(2.0 * available)

    return quad(momentum_width, -x_turn, x_turn, epsabs=1.0e-12)[0]


def exact_band_density(quartic_lambda: float) -> float:
    upper = phase_volume(ENERGY + ENERGY_BAND, quartic_lambda)
    lower = phase_volume(ENERGY - ENERGY_BAND, quartic_lambda)
    return (upper - lower) / (2.0 * ENERGY_BAND)


def liouville_band_density(
    epsilon: float, aspect: float, shear: float, quartic_lambda: float
) -> float:
    """Cell-center quadrature of invariant phase volume in a fixed energy band."""
    dq = epsilon * aspect
    dp = epsilon / aspect
    q = centers_for_width(-BOUND, BOUND, dq)
    p_extent = BOUND * (1.0 + abs(shear))
    p = centers_for_width(-p_extent, p_extent, dp)
    count = 0
    for q_value in q:
        physical_p = p - shear * q_value
        in_domain = np.abs(physical_p) <= BOUND
        energy = 0.5 * physical_p**2 + potential(q_value, quartic_lambda)
        in_band = np.abs(energy - ENERGY) <= ENERGY_BAND
        count += int(np.count_nonzero(in_domain & in_band))
    band_volume = count * dq * dp
    return band_volume / (2.0 * ENERGY_BAND)


def band_density_results() -> list[BandDensity]:
    rows = []
    schemes = (
        ("aspect=0.5", 0.5, 0.0),
        ("aspect=1", 1.0, 0.0),
        ("aspect=2", 2.0, 0.0),
        ("canonical shear=1", 1.0, SHEAR),
    )
    for name, quartic_lambda in (
        ("harmonic", 0.0),
        ("anharmonic", ANHARMONIC_LAMBDA),
    ):
        for scheme, aspect, shear in schemes:
            for epsilon in BAND_RESOLUTIONS:
                rows.append(
                    BandDensity(
                        hamiltonian=name,
                        scheme=scheme,
                        epsilon=epsilon,
                        density=liouville_band_density(
                            epsilon, aspect, shear, quartic_lambda
                        ),
                    )
                )
    return rows


def main() -> None:
    print("Noether-shell invariance gate")
    print()

    cell_rows = []
    for epsilon in CELL_RESOLUTIONS:
        for aspect in ASPECTS:
            cell_rows.append(rectangular_harmonic_count(epsilon, aspect))
        cell_rows.append(canonical_shear_count(epsilon))

    print("1. Raw intersecting-cell finite parts")
    print("   scheme                 epsilon   raw bits   finite bits")
    for row in cell_rows:
        print(
            f"   {row.scheme:22s} {row.epsilon:7.4f}   "
            f"{row.bits:8.4f}   {row.finite_bits:10.4f}"
        )
    finest_cell = [row for row in cell_rows if row.epsilon == CELL_RESOLUTIONS[-1]]
    finite_values = np.asarray([row.finite_bits for row in finest_cell])
    finite_spread = float(np.ptp(finite_values))
    print(f"   finest-grid finite-part spread: {finite_spread:.4f} bits")
    print()

    density_rows = band_density_results()
    print(f"2. Liouville band density, energy half-width={ENERGY_BAND}")
    density_errors = []
    scheme_spreads = []
    for name, quartic_lambda in (
        ("harmonic", 0.0),
        ("anharmonic", ANHARMONIC_LAMBDA),
    ):
        exact = exact_band_density(quartic_lambda)
        finest = [
            row
            for row in density_rows
            if row.hamiltonian == name and row.epsilon == BAND_RESOLUTIONS[-1]
        ]
        values = np.asarray([row.density for row in finest])
        errors = np.abs(values - exact) / exact
        density_errors.extend(errors.tolist())
        scheme_spreads.append(float(np.ptp(values) / np.mean(values)))
        print(f"   {name}: exact band density={exact:.6f}")
        for row, error in zip(finest, errors):
            print(
                f"     {row.scheme:22s} density={row.density:.6f}, "
                f"relative error={100.0 * error:.3f}%"
            )
        print(
            f"     cross-scheme spread={100.0 * scheme_spreads[-1]:.3f}%"
        )
    print()

    checks = {
        "raw finite part is coordinate/grid invariant": finite_spread < 0.10,
        "Liouville quadrature agrees with exact density": max(density_errors) < 0.02,
        "Liouville density is cross-scheme invariant": max(scheme_spreads) < 0.02,
        "second Hamiltonian passes measure test": all(
            error < 0.02
            for row, error in zip(
                [
                    row
                    for row in density_rows
                    if row.epsilon == BAND_RESOLUTIONS[-1]
                ],
                density_errors,
            )
            if row.hamiltonian == "anharmonic"
        ),
    }
    print("Invariance checks")
    for name, passed in checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(
        "INVARIANT CANDIDATE: "
        f"{'RAW CELL FINITE PART' if checks['raw finite part is coordinate/grid invariant'] else 'LIOUVILLE DENSITY ONLY'}"
    )


if __name__ == "__main__":
    main()
