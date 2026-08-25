"""Nonlinear, information-functional, and clock tests for history measure.

This extends ``run_history_pushforward.py`` in three directions:

1. An anharmonic oscillator is evolved independently in physical canonical
   coordinates and after a nonlinear canonical momentum shear.
2. Surviving-volume information and posterior KL information are compared as
   an observation record accumulates.
3. The same harmonic history functional is evaluated with three clock
   parameterizations, with and without the lapse/Jacobian required to
   represent the invariant physical-time integral.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc


SOBOL_POWER = 17
BOUND = 1.5
DURATION = 4.0
RECORD_SCALE = 0.35
ANHARMONIC_LAMBDA = 0.5
SHEAR = 0.2
ANHARMONIC_STEPS = (0.1, 0.05, 0.025, 0.0125)
CLOCK_SLICES = (20, 40, 80, 160, 320)


def initial_ensemble() -> tuple[np.ndarray, np.ndarray]:
    unit = qmc.Sobol(d=2, scramble=True, seed=20260825).random_base2(SOBOL_POWER)
    physical = -BOUND + 2.0 * BOUND * unit
    return physical[:, 0], physical[:, 1]


def statistics(log_weight: np.ndarray) -> tuple[float, float]:
    maximum = float(np.max(log_weight))
    shifted = np.exp(log_weight - maximum)
    log_z = maximum + math.log(float(np.mean(shifted)))
    posterior_log_weight = float(np.sum(shifted * log_weight) / np.sum(shifted))
    return -log_z / math.log(2.0), (posterior_log_weight - log_z) / math.log(2.0)


def force(x: np.ndarray) -> np.ndarray:
    return -x - ANHARMONIC_LAMBDA * x**3


def physical_midpoint_step(
    x: np.ndarray, p: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    x_mid = x + 0.5 * dt * p
    p_mid = p + 0.5 * dt * force(x)
    return x + dt * p_mid, p + dt * force(x_mid)


def transformed_derivative(q: np.ndarray, capital_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    physical_p = capital_p - SHEAR * q**3
    dq = physical_p
    dcapital_p = (
        3.0 * SHEAR * q**2 * physical_p
        - q
        - ANHARMONIC_LAMBDA * q**3
    )
    return dq, dcapital_p


def transformed_midpoint_step(
    q: np.ndarray, capital_p: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    dq, dp = transformed_derivative(q, capital_p)
    q_mid = q + 0.5 * dt * dq
    p_mid = capital_p + 0.5 * dt * dp
    dq_mid, dp_mid = transformed_derivative(q_mid, p_mid)
    return q + dt * dq_mid, capital_p + dt * dp_mid


def reference_anharmonic(times: np.ndarray) -> np.ndarray:
    def rhs(_time: float, state: np.ndarray) -> tuple[float, float]:
        x, p = state
        return p, -x - ANHARMONIC_LAMBDA * x**3

    solution = solve_ivp(
        rhs,
        (0.0, DURATION),
        (1.0, 0.0),
        t_eval=times,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def anharmonic_statistics(
    x0: np.ndarray, p0: np.ndarray, dt: float, transformed: bool
) -> tuple[float, float]:
    steps = int(round(DURATION / dt))
    times = dt * np.arange(1, steps + 1)
    record = reference_anharmonic(times)
    log_weight = np.zeros_like(x0)
    if transformed:
        q = x0.copy()
        capital_p = p0 + SHEAR * x0**3
        for observed in record:
            q, capital_p = transformed_midpoint_step(q, capital_p, dt)
            log_weight -= 0.5 * dt * ((q - observed) / RECORD_SCALE) ** 2
    else:
        x = x0.copy()
        p = p0.copy()
        for observed in record:
            x, p = physical_midpoint_step(x, p, dt)
            log_weight -= 0.5 * dt * ((x - observed) / RECORD_SCALE) ** 2
    return statistics(log_weight)


def harmonic_position(x0: np.ndarray, p0: np.ndarray, time: np.ndarray) -> np.ndarray:
    return x0[:, None] * np.cos(time)[None, :] + p0[:, None] * np.sin(time)[None, :]


def harmonic_record_statistics(
    x0: np.ndarray, p0: np.ndarray, duration: float, slices: int
) -> tuple[float, float]:
    dt = duration / slices
    times = dt * np.arange(1, slices + 1)
    log_weight = np.zeros_like(x0)
    for time in times:
        prediction = x0 * math.cos(time) + p0 * math.sin(time)
        observed = math.cos(time)
        log_weight -= 0.5 * dt * ((prediction - observed) / RECORD_SCALE) ** 2
    return statistics(log_weight)


def clock_map(name: str, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "linear":
        return DURATION * u, np.full_like(u, DURATION)
    if name == "quadratic":
        return DURATION * u**2, 2.0 * DURATION * u
    if name == "exponential":
        rate = 2.0
        denominator = math.exp(rate) - 1.0
        exponential = np.exp(rate * u)
        return DURATION * (exponential - 1.0) / denominator, (
            DURATION * rate * exponential / denominator
        )
    raise ValueError(name)


def clock_statistics(
    x0: np.ndarray, p0: np.ndarray, name: str, slices: int, invariant: bool
) -> tuple[float, float]:
    du = 1.0 / slices
    u = (np.arange(slices) + 0.5) * du
    times, lapse = clock_map(name, u)
    log_weight = np.zeros_like(x0)
    for time, local_lapse in zip(times, lapse):
        prediction = x0 * math.cos(float(time)) + p0 * math.sin(float(time))
        observed = math.cos(float(time))
        integration_weight = float(local_lapse) * du if invariant else DURATION * du
        log_weight -= (
            0.5 * integration_weight * ((prediction - observed) / RECORD_SCALE) ** 2
        )
    return statistics(log_weight)


def main() -> None:
    x0, p0 = initial_ensemble()
    print("Extended invariant-history-measure tests")
    print(f"ensemble: {x0.size:,} Sobol Liouville samples")
    print()

    print("1. Anharmonic dynamics under nonlinear canonical shear")
    print("dt       C(x,p)       C(Q,P)       |delta C|    KL(x,p)")
    anharmonic_rows = []
    for dt in ANHARMONIC_STEPS:
        physical = anharmonic_statistics(x0, p0, dt, transformed=False)
        transformed = anharmonic_statistics(x0, p0, dt, transformed=True)
        anharmonic_rows.append((dt, physical, transformed))
        print(
            f"{dt:0.4f}   {physical[0]:10.6f}   {transformed[0]:10.6f}   "
            f"{abs(physical[0] - transformed[0]):10.6f}   {physical[1]:10.6f}"
        )
    nonlinear_finest_error = abs(
        anharmonic_rows[-1][1][0] - anharmonic_rows[-1][2][0]
    )
    nonlinear_error_ratio = nonlinear_finest_error / abs(
        anharmonic_rows[-2][1][0] - anharmonic_rows[-2][2][0]
    )
    print()

    print("2. Information functionals as the harmonic record accumulates")
    print("duration   survival C bits   posterior KL bits   difference")
    information_rows = []
    for duration in (0.5, 1.0, 2.0, 3.0, 4.0):
        survival, kl = harmonic_record_statistics(
            x0, p0, duration=duration, slices=int(round(duration / 0.0125))
        )
        information_rows.append((duration, survival, kl))
        print(f"{duration:5.1f}       {survival:10.6f}         {kl:10.6f}     {survival-kl:10.6f}")
    survival_monotone = all(
        later[1] >= earlier[1]
        for earlier, later in zip(information_rows, information_rows[1:])
    )
    kl_monotone = all(
        later[2] >= earlier[2]
        for earlier, later in zip(information_rows, information_rows[1:])
    )
    print()

    print("3. Clock reparameterization at 320 slices")
    print("clock          invariant C   invariant KL   naive C")
    clock_names = ("linear", "quadratic", "exponential")
    clock_rows = []
    for name in clock_names:
        invariant_result = clock_statistics(x0, p0, name, CLOCK_SLICES[-1], True)
        naive_result = clock_statistics(x0, p0, name, CLOCK_SLICES[-1], False)
        clock_rows.append((name, invariant_result, naive_result))
        print(
            f"{name:11s}   {invariant_result[0]:11.7f}   "
            f"{invariant_result[1]:12.7f}   {naive_result[0]:9.6f}"
        )
    invariant_values = np.asarray([row[1][0] for row in clock_rows])
    naive_values = np.asarray([row[2][0] for row in clock_rows])
    invariant_spread = float(np.ptp(invariant_values))
    naive_spread = float(np.ptp(naive_values))

    quadratic_convergence = []
    for slices in CLOCK_SLICES:
        quadratic_convergence.append(
            clock_statistics(x0, p0, "quadratic", slices, True)[0]
        )
    clock_refinement_change = abs(quadratic_convergence[-1] - quadratic_convergence[-2])
    print()
    print("Checks")
    checks = {
        "nonlinear canonical discrepancy decreases under refinement": nonlinear_error_ratio < 0.35,
        "nonlinear canonical agreement at finest step": nonlinear_finest_error < 0.002,
        "surviving-volume information is monotone": survival_monotone,
        "posterior KL is monotone for this record": kl_monotone,
        "lapse-weighted clock invariance": invariant_spread < 0.001,
        "clock-refinement convergence": clock_refinement_change < 0.001,
        "naive slice weighting fails clock invariance": naive_spread > 0.05,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print()
    print(f"finest nonlinear canonical discrepancy: {nonlinear_finest_error:.6f} bits")
    print(f"invariant cross-clock spread: {invariant_spread:.6f} bits")
    print(f"naive cross-clock spread: {naive_spread:.6f} bits")
    print(f"quadratic-clock finest refinement change: {clock_refinement_change:.6f} bits")
    print(f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
