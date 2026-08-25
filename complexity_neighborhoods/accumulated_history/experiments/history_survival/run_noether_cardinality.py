"""Conservation- and action-based compatible-history cardinality tests.

No geometric tolerance is fitted. A state cell is energy-admissible exactly
when the cell intersects a fixed Hamiltonian level set. A path update is
action-admissible exactly when the interval of its three discretization cells
contains a solution of the discrete Euler-Lagrange equation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SPATIAL_RESOLUTIONS = (0.2, 0.1, 0.05, 0.025, 0.0125)
ENERGY_BOUND = 2.0
ENERGY_LEVEL = 1.0
ACTION_BOUND = 1.5
REFERENCE_DT = 0.2
PHYSICAL_DURATION = 2.4
TIME_STEPS = (0.4, 0.2, 0.1)
REFERENCE_ACTION_RESOLUTION = 0.025
OMEGA = 1.0


@dataclass(frozen=True)
class ShellResult:
    epsilon: float
    total_cells: int
    admissible_cells: int
    fraction: float
    bits: float
    renormalized_bits: float


@dataclass(frozen=True)
class ActionResult:
    epsilon: float
    dt: float
    updates: int
    survival: float
    bits: float
    bits_per_update: float
    bits_per_time: float
    renormalized_bits_per_update: float


def cell_centers(bound: float, epsilon: float) -> np.ndarray:
    """Centers of equal cells that exactly tile [-bound, bound]."""
    count = int(round(2.0 * bound / epsilon))
    if not math.isclose(count * epsilon, 2.0 * bound, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Resolution must tile the bounded domain exactly")
    return -bound + 0.5 * epsilon + epsilon * np.arange(count)


def squared_interval_extrema(centers: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    lower = centers - 0.5 * epsilon
    upper = centers + 0.5 * epsilon
    minimum_absolute = np.where(
        (lower <= 0.0) & (upper >= 0.0),
        0.0,
        np.minimum(np.abs(lower), np.abs(upper)),
    )
    maximum_absolute = np.maximum(np.abs(lower), np.abs(upper))
    return minimum_absolute**2, maximum_absolute**2


def energy_shell_count(epsilon: float) -> ShellResult:
    """Count phase-space cells intersecting H=(x^2+p^2)/2=E."""
    centers = cell_centers(ENERGY_BOUND, epsilon)
    minimum_squared, maximum_squared = squared_interval_extrema(centers, epsilon)
    h_min = 0.5 * (minimum_squared[:, None] + minimum_squared[None, :])
    h_max = 0.5 * (maximum_squared[:, None] + maximum_squared[None, :])
    admissible = (h_min <= ENERGY_LEVEL) & (h_max >= ENERGY_LEVEL)
    total = admissible.size
    count = int(np.count_nonzero(admissible))
    fraction = count / total
    bits = -math.log2(fraction)
    # A codimension-one shell necessarily contributes one log-resolution term.
    renormalized = bits - math.log2(1.0 / epsilon)
    return ShellResult(epsilon, total, count, fraction, bits, renormalized)


def containing_cell_index(value: float, bound: float, epsilon: float, count: int) -> int:
    index = int(math.floor((value + bound) / epsilon))
    return min(max(index, 0), count - 1)


def action_path_count(epsilon: float, dt: float) -> ActionResult:
    """Count paths compatible with the discrete oscillator action.

    For L=(xdot^2-omega^2*x^2)/2, the central discrete Euler-Lagrange
    residual is x_next + (-2+dt^2*omega^2)x_now + x_previous. Cell interval
    arithmetic fixes the admissibility width; no residual tolerance is fitted.
    """
    grid = cell_centers(ACTION_BOUND, epsilon)
    count = len(grid)
    coefficient = -2.0 + (dt * OMEGA) ** 2
    residual_half_width = 0.5 * epsilon * (2.0 + abs(coefficient))

    initial_position = 1.0
    first_position = math.cos(OMEGA * dt)
    previous_index = containing_cell_index(
        initial_position, ACTION_BOUND, epsilon, count
    )
    current_index = containing_cell_index(first_position, ACTION_BOUND, epsilon, count)
    weights: dict[tuple[int, int], float] = {(previous_index, current_index): 1.0}
    updates = int(round(PHYSICAL_DURATION / dt)) - 1

    first_center = float(grid[0])
    for _ in range(updates):
        next_weights: dict[tuple[int, int], float] = {}
        for (previous, current), probability in weights.items():
            target = -coefficient * grid[current] - grid[previous]
            lower = target - residual_half_width
            upper = target + residual_half_width
            first_candidate = max(
                0, int(math.ceil((lower - first_center) / epsilon - 1.0e-12))
            )
            last_candidate = min(
                count - 1,
                int(math.floor((upper - first_center) / epsilon + 1.0e-12)),
            )
            if first_candidate > last_candidate:
                continue
            transition_probability = probability / count
            for candidate in range(first_candidate, last_candidate + 1):
                key = (current, candidate)
                next_weights[key] = next_weights.get(key, 0.0) + transition_probability
        weights = next_weights

    survival = float(sum(weights.values()))
    if survival <= 0.0:
        raise ValueError("No action-compatible paths survived")
    bits = -math.log2(survival)
    bits_per_update = bits / updates
    renormalized = bits_per_update - math.log2(1.0 / epsilon)
    return ActionResult(
        epsilon=epsilon,
        dt=dt,
        updates=updates,
        survival=survival,
        bits=bits,
        bits_per_update=bits_per_update,
        bits_per_time=bits / PHYSICAL_DURATION,
        renormalized_bits_per_update=renormalized,
    )


def relative_span(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.ptp(array) / abs(np.mean(array)))


def main() -> None:
    print("Noether/action compatible-history cardinality")
    print("Hamiltonian: H=(x^2+p^2)/2; no fitted tolerance")
    print()

    shell_results = [energy_shell_count(epsilon) for epsilon in SPATIAL_RESOLUTIONS]
    print("1. Energy-shell cell cardinality")
    print("   epsilon    admissible/total       raw bits   codimension-renormalized")
    for result in shell_results:
        print(
            f"   {result.epsilon:7.4f}   {result.admissible_cells:7d}/{result.total_cells:<7d} "
            f"{result.bits:10.4f}   {result.renormalized_bits:10.4f}"
        )
    print()

    action_results = [
        action_path_count(epsilon, REFERENCE_DT)
        for epsilon in SPATIAL_RESOLUTIONS
    ]
    print(f"2. Stationary-action histories at dt={REFERENCE_DT}")
    print("   epsilon    survival          bits/update   renormalized/update")
    for result in action_results:
        print(
            f"   {result.epsilon:7.4f}   {result.survival:12.4e}   "
            f"{result.bits_per_update:11.4f}   "
            f"{result.renormalized_bits_per_update:11.4f}"
        )
    print()

    time_results = [
        action_path_count(REFERENCE_ACTION_RESOLUTION, dt) for dt in TIME_STEPS
    ]
    print(
        "3. Fixed-duration action histories under time refinement "
        f"(epsilon={REFERENCE_ACTION_RESOLUTION})"
    )
    print("   dt       updates   total bits   bits/unit time")
    for result in time_results:
        print(
            f"   {result.dt:6.3f}   {result.updates:7d}   "
            f"{result.bits:10.4f}   {result.bits_per_time:14.4f}"
        )
    print()

    shell_raw = [result.bits for result in shell_results[-3:]]
    shell_renormalized = [result.renormalized_bits for result in shell_results[-3:]]
    action_raw = [result.bits_per_update for result in action_results[-3:]]
    action_renormalized = [
        result.renormalized_bits_per_update for result in action_results[-3:]
    ]
    time_rates = [result.bits_per_time for result in time_results]
    checks = {
        "energy-shell raw bits converge": relative_span(shell_raw) < 0.10,
        "energy-shell codimension term converges": relative_span(shell_renormalized) < 0.10,
        "action raw bits/update converge": relative_span(action_raw) < 0.10,
        "action renormalized bits/update converge": relative_span(action_renormalized) < 0.10,
        "absolute bits/time converge": relative_span(time_rates) < 0.10,
    }
    print("Resolution-identifiability checks")
    for name, passed in checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print()
    absolute_identifiable = (
        checks["energy-shell raw bits converge"]
        and checks["action raw bits/update converge"]
        and checks["absolute bits/time converge"]
    )
    print(
        "ABSOLUTE RATE IDENTIFIABLE: "
        f"{'YES' if absolute_identifiable else 'NO'}"
    )


if __name__ == "__main__":
    main()
