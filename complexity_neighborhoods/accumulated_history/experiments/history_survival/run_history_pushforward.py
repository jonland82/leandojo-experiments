"""Test a Liouville-induced measure on complete deterministic histories.

The initial Liouville ensemble is pushed through exact harmonic dynamics.  A
soft compatibility functional represents a continuous observation record.
The experiment checks canonical-coordinate invariance and convergence under
temporal refinement.  A fixed-observation hard survival set is included as a
direct analogue of the accumulated-history papers' rho.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc


BOUND = 2.0
TRUE_INITIAL = (1.0, 0.0)
DURATION = 4.0
RECORD_SCALE = 0.35
TIME_STEPS = (0.2, 0.1, 0.05, 0.025, 0.0125)
SOBOL_POWER = 19
HARD_TIMES = np.asarray((0.8, 1.6, 2.4, 3.2, 4.0))
HARD_TOLERANCE = 0.15


def liouville_initial_ensemble() -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight points in a bounded phase-space box."""
    unit = qmc.Sobol(d=2, scramble=True, seed=20260825).random_base2(SOBOL_POWER)
    physical = -BOUND + 2.0 * BOUND * unit
    return physical[:, 0], physical[:, 1]


def physical_position(x0: np.ndarray, p0: np.ndarray, time: float) -> np.ndarray:
    return x0 * math.cos(time) + p0 * math.sin(time)


def sheared_position(q0: np.ndarray, capital_p0: np.ndarray, time: float) -> np.ndarray:
    """Position evolved in Q=x, P=p+x canonical coordinates."""
    return (math.cos(time) - math.sin(time)) * q0 + math.sin(time) * capital_p0


def true_position(time: float) -> float:
    return TRUE_INITIAL[0] * math.cos(time) + TRUE_INITIAL[1] * math.sin(time)


def record_statistics(
    x0: np.ndarray, p0: np.ndarray, dt: float, sheared: bool
) -> tuple[float, float]:
    """Return -log2 mean compatibility and posterior KL information."""
    log_weight = np.zeros_like(x0)
    capital_p0 = p0 + x0
    steps = int(round(DURATION / dt))
    for index in range(1, steps + 1):
        time = index * dt
        if sheared:
            prediction = sheared_position(x0, capital_p0, time)
        else:
            prediction = physical_position(x0, p0, time)
        residual = (prediction - true_position(time)) / RECORD_SCALE
        log_weight -= 0.5 * dt * residual**2

    maximum = float(np.max(log_weight))
    shifted = np.exp(log_weight - maximum)
    mean_weight = math.exp(maximum) * float(np.mean(shifted))
    log_mean_weight = math.log(mean_weight)
    posterior_mean_log_weight = float(np.sum(shifted * log_weight) / np.sum(shifted))
    survival_bits = -log_mean_weight / math.log(2.0)
    kl_bits = (posterior_mean_log_weight - log_mean_weight) / math.log(2.0)
    return survival_bits, kl_bits


def hard_survival(x0: np.ndarray, p0: np.ndarray, sheared: bool) -> tuple[float, float]:
    surviving = np.ones(x0.shape, dtype=bool)
    capital_p0 = p0 + x0
    for time in HARD_TIMES:
        if sheared:
            prediction = sheared_position(x0, capital_p0, float(time))
        else:
            prediction = physical_position(x0, p0, float(time))
        surviving &= np.abs(prediction - true_position(float(time))) <= HARD_TOLERANCE
    rho = float(np.mean(surviving))
    return rho, -math.log2(rho)


def main() -> None:
    x0, p0 = liouville_initial_ensemble()
    print("Liouville pushforward history-measure test")
    print(f"ensemble: {x0.size:,} Sobol points on [-{BOUND:g},{BOUND:g}]^2")
    print()
    print("Continuous-record compatibility")
    print("dt       C(x,p) bits   C(Q,P) bits   KL(x,p) bits  |delta C|")
    rows = []
    for dt in TIME_STEPS:
        physical = record_statistics(x0, p0, dt, sheared=False)
        sheared = record_statistics(x0, p0, dt, sheared=True)
        rows.append((dt, physical, sheared))
        print(
            f"{dt:0.4f}   {physical[0]:12.7f}  {sheared[0]:12.7f}  "
            f"{physical[1]:13.7f}  {abs(physical[0] - sheared[0]):.3e}"
        )

    finest_c = rows[-1][1][0]
    preceding_c = rows[-2][1][0]
    refinement_change = abs(finest_c - preceding_c)
    coordinate_error = max(abs(a[0] - b[0]) for _, a, b in rows)

    rho_physical, bits_physical = hard_survival(x0, p0, sheared=False)
    rho_sheared, bits_sheared = hard_survival(x0, p0, sheared=True)
    print()
    print("Fixed-observation hard survival")
    print(
        f"rho(x,p)={rho_physical:.8f}, rho(Q,P)={rho_sheared:.8f}, "
        f"C={bits_physical:.6f} bits"
    )
    print()
    checks = {
        "canonical-coordinate invariance": coordinate_error < 1.0e-12,
        "temporal-refinement convergence": refinement_change < 0.01,
        "hard-set coordinate invariance": abs(rho_physical - rho_sheared) < 1.0e-15,
        "nontrivial history information": finest_c > 0.0 and rows[-1][1][1] > 0.0,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"finest-step change in C: {refinement_change:.6f} bits")
    print(f"maximum canonical-coordinate discrepancy: {coordinate_error:.3e} bits")
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
