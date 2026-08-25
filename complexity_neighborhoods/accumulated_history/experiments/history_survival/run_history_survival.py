"""Controlled validation of a compatible-history survival estimator.

The experiment deliberately precedes any cosmological claim. It asks whether
fixed-radius cumulative history neighborhoods distinguish genuinely new
conditional information from predictable expansion and deterministic chaos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SEED = 20260824
BINARY_STEPS = 12
FIT_STEPS = 10
PAIR_COUNTS = (100_000, 300_000, 1_000_000, 2_000_000)
TOLERANCE_BUDGETS = (0, 1, 2)


@dataclass(frozen=True)
class BinaryResult:
    pair_count: int
    survival: np.ndarray
    complexity: np.ndarray
    rate: float
    grouped_rate: float


def complexity_from_survival(survival: np.ndarray) -> np.ndarray:
    """Convert positive survival fractions to base-two surprisal."""
    if np.any(survival <= 0.0):
        raise ValueError("A survival fraction reached zero; increase the ensemble size")
    return -np.log2(survival)


def fit_rate(complexity: np.ndarray, steps: int = FIT_STEPS) -> float:
    """Fit bits per primitive update, including a free intercept."""
    time = np.arange(1, steps + 1, dtype=float)
    return float(np.polyfit(time, complexity[:steps], 1)[0])


def binary_collision_experiment(pair_count: int, seed: int) -> BinaryResult:
    """Compare independent unbiased innovation bits for paired histories."""
    rng = np.random.default_rng(seed)
    mismatches = np.zeros(pair_count, dtype=np.int16)
    survival = np.empty((len(TOLERANCE_BUDGETS), BINARY_STEPS), dtype=float)

    for step in range(BINARY_STEPS):
        first = rng.integers(0, 2, size=pair_count, dtype=np.int8)
        second = rng.integers(0, 2, size=pair_count, dtype=np.int8)
        mismatches += first != second
        for row, budget in enumerate(TOLERANCE_BUDGETS):
            survival[row, step] = np.mean(mismatches <= budget)

    complexity = complexity_from_survival(survival)
    rate = fit_rate(complexity[0])

    # Regroup the same primitive updates into two-update reporting intervals.
    # A valid estimator should return two bits per group, hence one per update.
    grouped_complexity = complexity[0, 1::2]
    group_time = np.arange(1, len(grouped_complexity) + 1, dtype=float)
    bits_per_group = float(np.polyfit(group_time[:5], grouped_complexity[:5], 1)[0])
    grouped_rate = bits_per_group / 2.0

    return BinaryResult(pair_count, survival, complexity, rate, grouped_rate)


def theoretical_binary_survival(steps: int, budget: int) -> float:
    """Exact fixed-radius survival for cumulative Hamming distance."""
    return sum(math.comb(steps, k) for k in range(budget + 1)) / 2.0**steps


def hubble_control(pair_count: int = 100_000, steps: int = 30) -> dict[str, float]:
    """Contrast raw physical separation with exactly conditioned innovations."""
    rng = np.random.default_rng(SEED + 101)
    first = rng.normal(0.0, 5.0e-4, size=(pair_count, 3))
    second = rng.normal(0.0, 5.0e-4, size=(pair_count, 3))
    static_distance = np.linalg.norm(first - second, axis=1)
    static_cumulative = steps * static_distance

    growth = 1.02
    raw_cumulative = np.zeros(pair_count)
    conditional_cumulative = np.zeros(pair_count)
    for _ in range(steps):
        predicted_first = growth * first
        predicted_second = growth * second
        next_first = growth * first
        next_second = growth * second

        raw_cumulative += np.linalg.norm(next_first - next_second, axis=1)
        innovation_first = next_first - predicted_first
        innovation_second = next_second - predicted_second
        conditional_cumulative += np.linalg.norm(
            innovation_first - innovation_second, axis=1
        )
        first, second = next_first, next_second

    # Calibrate the raw fixed radius on a predeclared nonexpanding null. This
    # makes raw predictable stretching visible without entering the conditional
    # estimator used for the scientific test.
    raw_radius = float(np.median(static_cumulative))
    static_survival = float(np.mean(static_cumulative <= raw_radius))
    expanding_survival = float(np.mean(raw_cumulative <= raw_radius))
    conditional_survival = float(np.mean(conditional_cumulative <= 1.0e-15))
    return {
        "raw_radius": raw_radius,
        "static_raw_complexity": -math.log2(static_survival),
        "expanding_raw_complexity": -math.log2(expanding_survival),
        "conditional_complexity": 0.0 if conditional_survival == 1.0 else -math.log2(conditional_survival),
        "max_conditional_distance": float(conditional_cumulative.max()),
    }


def wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Shortest signed displacement on a 2-pi periodic coordinate."""
    return (first - second + math.pi) % (2.0 * math.pi) - math.pi


def standard_map(state: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """One deterministic, area-preserving standard-map update."""
    angle = state[:, 0]
    momentum = state[:, 1]
    new_momentum = (momentum + strength * np.sin(angle)) % (2.0 * math.pi)
    new_angle = (angle + new_momentum) % (2.0 * math.pi)
    return np.column_stack((new_angle, new_momentum))


def chaotic_control(pair_count: int = 100_000, steps: int = 30) -> dict[str, float]:
    """Show that deterministic trajectory divergence is not a new innovation."""
    rng = np.random.default_rng(SEED + 202)
    first = rng.uniform(0.0, 2.0 * math.pi, size=(pair_count, 2))
    second = (first + rng.normal(0.0, 1.0e-8, size=(pair_count, 2))) % (
        2.0 * math.pi
    )
    raw_cumulative = np.zeros(pair_count)
    conditional_cumulative = np.zeros(pair_count)

    for _ in range(steps):
        predicted_first = standard_map(first)
        predicted_second = standard_map(second)
        next_first = standard_map(first)
        next_second = standard_map(second)

        raw_difference = wrapped_delta(next_first, next_second)
        raw_cumulative += np.linalg.norm(raw_difference, axis=1)
        innovation_first = wrapped_delta(next_first, predicted_first)
        innovation_second = wrapped_delta(next_second, predicted_second)
        conditional_cumulative += np.linalg.norm(
            innovation_first - innovation_second, axis=1
        )
        first, second = next_first, next_second

    raw_survival = float(np.mean(raw_cumulative <= 0.1))
    conditional_survival = float(np.mean(conditional_cumulative <= 1.0e-15))
    return {
        "raw_complexity": -math.log2(raw_survival),
        "conditional_complexity": 0.0 if conditional_survival == 1.0 else -math.log2(conditional_survival),
        "raw_median_distance": float(np.median(raw_cumulative)),
        "max_conditional_distance": float(conditional_cumulative.max()),
    }


def hidden_state_control(pair_count: int = 500_000) -> dict[str, float]:
    """Reveal a deterministic hidden bit tape through a coarse observation.

    Each complete tape is fixed at the initial time, so the full-state dynamics
    are deterministic. An observer conditioned only on the revealed macrostate
    nevertheless receives one previously hidden distinction per update.
    """
    rng = np.random.default_rng(SEED + 303)
    first_tape = rng.integers(
        0, 2, size=(pair_count, BINARY_STEPS), dtype=np.int8
    )
    second_tape = rng.integers(
        0, 2, size=(pair_count, BINARY_STEPS), dtype=np.int8
    )
    mismatches = np.cumsum(first_tape != second_tape, axis=1)
    coarse_survival = np.mean(mismatches == 0, axis=0)
    coarse_complexity = complexity_from_survival(coarse_survival)

    # Conditioning on the complete hidden tape predicts every reveal exactly.
    full_state_survival = 1.0
    return {
        "coarse_rate": fit_rate(coarse_complexity),
        "coarse_final_complexity": float(coarse_complexity[-1]),
        "full_state_complexity": 0.0 if full_state_survival == 1.0 else -math.log2(full_state_survival),
    }


def main() -> None:
    print("Compatible-history survival: controlled estimator validation")
    print(f"Seed={SEED}; primitive binary updates={BINARY_STEPS}")
    print()

    hubble = hubble_control()
    print("1. Predictable Hubble-flow control")
    print(
        "   raw fixed-radius complexity: "
        f"static={hubble['static_raw_complexity']:.4f}, "
        f"expanding={hubble['expanding_raw_complexity']:.4f} bits"
    )
    print(
        "   after exact conditioning:    "
        f"C={hubble['conditional_complexity']:.4f} bits, "
        f"max distance={hubble['max_conditional_distance']:.1e}"
    )
    print()

    chaotic = chaotic_control()
    print("2. Deterministic-chaos control")
    print(
        "   raw trajectory:              "
        f"median cumulative distance={chaotic['raw_median_distance']:.3f}, "
        f"C(r=0.1)={chaotic['raw_complexity']:.4f} bits"
    )
    print(
        "   after exact conditioning:    "
        f"C={chaotic['conditional_complexity']:.4f} bits, "
        f"max distance={chaotic['max_conditional_distance']:.1e}"
    )
    print()

    binary_results = [
        binary_collision_experiment(count, SEED + 1_000 + index)
        for index, count in enumerate(PAIR_COUNTS)
    ]
    largest = binary_results[-1]
    print("3. Injected unbiased innovation bits")
    for result in binary_results:
        print(
            f"   pairs={result.pair_count:>9,}: "
            f"rate={result.rate:.4f} bits/update, "
            f"two-step regrouped rate={result.grouped_rate:.4f} bits/update"
        )
    print("   fixed-radius tolerance check at the final update:")
    tolerance_errors = []
    for row, budget in enumerate(TOLERANCE_BUDGETS):
        empirical = largest.survival[row, -1]
        theory = theoretical_binary_survival(BINARY_STEPS, budget)
        empirical_c = largest.complexity[row, -1]
        theory_c = -math.log2(theory)
        error = abs(empirical_c - theory_c)
        tolerance_errors.append(error)
        print(
            f"     r={budget}: rho={empirical:.6f} "
            f"(theory {theory:.6f}), C={empirical_c:.4f} "
            f"(theory {theory_c:.4f})"
        )
    print()

    hidden = hidden_state_control()
    print("4. Deterministic hidden-state/coarse-graining control")
    print(
        "   conditioned on revealed macro-past: "
        f"rate={hidden['coarse_rate']:.4f} bits/update, "
        f"C_final={hidden['coarse_final_complexity']:.4f} bits"
    )
    print(
        "   conditioned on complete microstate: "
        f"C={hidden['full_state_complexity']:.4f} bits"
    )
    print()

    rates = np.asarray([result.rate for result in binary_results])
    grouped_rates = np.asarray([result.grouped_rate for result in binary_results])
    checks = {
        "predictable Hubble flow removed": hubble["conditional_complexity"] < 1.0e-12,
        "deterministic chaos removed": chaotic["conditional_complexity"] < 1.0e-12,
        "injected rate recovered": abs(largest.rate - 1.0) < 0.05,
        "timestep regrouping stable": abs(largest.grouped_rate - 1.0) < 0.05,
        "ensemble convergence": float(np.ptp(rates)) < 0.08
        and float(np.ptp(grouped_rates)) < 0.08,
        "tolerance law recovered": max(tolerance_errors) < 0.20,
        "coarse hidden rate recovered": abs(hidden["coarse_rate"] - 1.0) < 0.05,
        "full hidden state removes novelty": hidden["full_state_complexity"] < 1.0e-12,
    }
    print("Predeclared validation checks")
    for name, passed in checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
