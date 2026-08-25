"""Covariant causal-history diagnostic in a flat FLRW background.

For an observation event at scale factor a_o, compute the four-volume of its
causal past since the Big Bang,

    V4 = (4 pi / 3) integral dt a(t)^3 chi(t, t_o)^3,

where chi is the radial null-geodesic distance.  Then test whether a single
constant can relate expansion theta=3H to either d(log2 V4)/dt or dV4/dt.

The background is the best-fit two-channel expansion from the DESI shape test.
This is an internal consistency test, not an independent measurement of causal
complexity.
"""

from __future__ import annotations

import math

import numpy as np


OMEGA_M = 0.2975
OMEGA_H = 1.0 - OMEGA_M
REDSHIFTS = (0.0, 0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330)


def e_of_a(a: np.ndarray) -> np.ndarray:
    return np.sqrt(OMEGA_M / a**3 + OMEGA_H)


def dimensionless_age(a: float) -> float:
    """H0*t(a) for flat matter plus constant-history expansion."""
    return 2.0 / (3.0 * math.sqrt(OMEGA_H)) * math.asinh(
        math.sqrt(OMEGA_H / OMEGA_M) * a ** 1.5
    )


def causal_volume(a_observer: float, samples: int = 40_001) -> tuple[float, float]:
    """Return dimensionless V4 and dV4/d(H0*t_observer)."""
    # Quadratic spacing resolves the integrable early-time endpoint.
    x = np.linspace(1.0e-5, 1.0, samples)
    a = a_observer * x**2
    e = e_of_a(a)

    conformal_integrand = 1.0 / (a**2 * e)
    da = np.diff(a)
    conformal_steps = 0.5 * (conformal_integrand[:-1] + conformal_integrand[1:]) * da
    chi = np.empty_like(a)
    chi[-1] = 0.0
    chi[:-1] = np.cumsum(conformal_steps[::-1])[::-1]

    time_measure = a**2 / e
    volume_integrand = time_measure * chi**3
    area_integrand = time_measure * chi**2
    volume = 4.0 * math.pi / 3.0 * float(np.trapezoid(volume_integrand, a))
    derivative = 4.0 * math.pi / a_observer * float(np.trapezoid(area_integrand, a))
    return volume, derivative


def main() -> None:
    rows = []
    for z in REDSHIFTS:
        a = 1.0 / (1.0 + z)
        e = float(e_of_a(np.asarray(a)))
        age = dimensionless_age(a)
        volume, derivative = causal_volume(a)
        dlog2v_dtau = derivative / (volume * math.log(2.0))

        # Couplings required by 3H = k*dC/dt.  H0 cancels in k_log.
        k_log = 3.0 * e / dlog2v_dtau
        k_raw = 3.0 * e / derivative
        rows.append((z, a, age, volume, dlog2v_dtau, k_log, k_raw))

    present_log = rows[0][5]
    present_raw = rows[0][6]

    print("Relativistic causal-past history test")
    print(f"Background: flat two-channel fit, Omega_m={OMEGA_M:.4f}")
    print("A constant-coupling mechanism requires the last columns to equal 1.")
    print()
    print("   z      H0*t       V4*H0^4    k_log/k_log(0)   k_raw/k_raw(0)")
    for z, _a, age, volume, _rate, k_log, k_raw in rows:
        print(f"{z:5.3f}   {age:7.4f}   {volume:11.5e}       {k_log/present_log:8.4f}          {k_raw/present_raw:8.4f}")

    log_drift = max(row[5] for row in rows) / min(row[5] for row in rows)
    raw_drift = max(row[6] for row in rows) / min(row[6] for row in rows)
    print()
    print(f"Required log-volume coupling drift across range: {log_drift:.3f}x")
    print(f"Required raw-volume coupling drift across range: {raw_drift:.3f}x")
    print(f"Present log-volume coupling k_log:              {present_log:.4f}")


if __name__ == "__main__":
    main()
