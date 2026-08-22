"""Test fixed complexity/history laws against the DESI DR2 BAO data vector.

The BAO ruler calibration X = c/(H0 r_d) is fitted analytically for every
model.  Consequently the fixed-p comparisons test redshift-dependent shape,
not the Hubble normalization that motivated the hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


C_KM_S = 299_792.458
R_D_MPC = 147.09  # reference early-Universe ruler, used only to display H0
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Datum:
    z: float
    value: float
    quantity: str


def load_data() -> tuple[list[Datum], np.ndarray]:
    rows: list[Datum] = []
    with (HERE / "data" / "desi_dr2_bao_mean.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            z, value, quantity = line.split()
            rows.append(Datum(float(z), float(value), quantity))
    covariance = np.loadtxt(HERE / "data" / "desi_dr2_bao_cov.txt")
    if covariance.shape != (len(rows), len(rows)):
        raise ValueError("BAO data vector and covariance have incompatible shapes")
    return rows, covariance


def integral_simpson(function, upper: float, panels: int = 2000) -> float:
    if panels % 2:
        panels += 1
    x = np.linspace(0.0, upper, panels + 1)
    y = function(x)
    return float((upper / panels) / 3.0 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def power_law_geometry(z: float, p: float) -> tuple[float, float]:
    """Return dimensionless (D_M H0/c, D_H H0/c) for a proportional to t^p."""
    exponent = 1.0 / p
    dh = (1.0 + z) ** (-exponent)
    if abs(exponent - 1.0) < 1.0e-12:
        dm = np.log1p(z)
    else:
        dm = ((1.0 + z) ** (1.0 - exponent) - 1.0) / (1.0 - exponent)
    return float(dm), float(dh)


def lcdm_geometry(z: float, omega_m: float = 0.3027) -> tuple[float, float]:
    def inverse_e(x: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(omega_m * (1.0 + x) ** 3 + 1.0 - omega_m)

    return integral_simpson(inverse_e, z), float(inverse_e(np.asarray(z)))


def prediction_shape(rows: list[Datum], geometry) -> np.ndarray:
    result = []
    for row in rows:
        dm, dh = geometry(row.z)
        if row.quantity == "DM_over_rs":
            result.append(dm)
        elif row.quantity == "DH_over_rs":
            result.append(dh)
        elif row.quantity == "DV_over_rs":
            result.append((row.z * dm * dm * dh) ** (1.0 / 3.0))
        else:
            raise ValueError(f"Unknown BAO quantity: {row.quantity}")
    return np.asarray(result)


def fit_scale(y: np.ndarray, covariance: np.ndarray, shape: np.ndarray) -> tuple[float, float]:
    precision_y = np.linalg.solve(covariance, y)
    precision_shape = np.linalg.solve(covariance, shape)
    scale = float(shape @ precision_y / (shape @ precision_shape))
    residual = y - scale * shape
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    return scale, chi2


def fit_power(rows: list[Datum], y: np.ndarray, covariance: np.ndarray) -> tuple[float, float, float]:
    # Dense deterministic profile search is sufficient for this one-dimensional test.
    candidates = np.linspace(0.35, 2.0, 16_501)
    best = (float("inf"), 0.0, 0.0)
    for p in candidates:
        shape = prediction_shape(rows, lambda z, trial=p: power_law_geometry(z, trial))
        scale, chi2 = fit_scale(y, covariance, shape)
        if chi2 < best[0]:
            best = (chi2, float(p), scale)
    return best[1], best[2], best[0]


def fit_two_channel(rows: list[Datum], y: np.ndarray, covariance: np.ndarray) -> tuple[float, float, float]:
    """Fit the matter/history mixture while profiling out the BAO scale."""
    candidates = np.linspace(0.05, 0.60, 1_101)
    best = (float("inf"), 0.0, 0.0)
    for omega_m in candidates:
        shape = prediction_shape(rows, lambda z, trial=omega_m: lcdm_geometry(z, trial))
        scale, chi2 = fit_scale(y, covariance, shape)
        if chi2 < best[0]:
            best = (chi2, float(omega_m), scale)
    return best[1], best[2], best[0]


def report_fixed(name: str, rows: list[Datum], y: np.ndarray, covariance: np.ndarray, geometry) -> None:
    shape = prediction_shape(rows, geometry)
    scale, chi2 = fit_scale(y, covariance, shape)
    h0 = C_KM_S / (scale * R_D_MPC)
    print(f"{name:30s} chi2={chi2:8.2f}  dof={len(y)-1:2d}  X={scale:7.3f}  H0(rd_ref)={h0:6.2f}")


def main() -> None:
    rows, covariance = load_data()
    y = np.asarray([row.value for row in rows])

    print("DESI DR2 BAO shape test")
    print("X=c/(H0*r_d) is fitted; fixed-p results therefore do not use H0.")
    print()
    report_fixed("matter-history: p=2/3", rows, y, covariance, lambda z: power_law_geometry(z, 2.0 / 3.0))
    report_fixed("spatial-history: p=1", rows, y, covariance, lambda z: power_law_geometry(z, 1.0))
    report_fixed("3D+time history: p=4/3", rows, y, covariance, lambda z: power_law_geometry(z, 4.0 / 3.0))
    report_fixed("flat LambdaCDM: Om=0.3027", rows, y, covariance, lcdm_geometry)

    p, scale, chi2 = fit_power(rows, y, covariance)
    h0 = C_KM_S / (scale * R_D_MPC)
    print(f"free constant-p history          chi2={chi2:8.2f}  dof={len(y)-2:2d}  p={p:7.4f}  H0(rd_ref)={h0:6.2f}")
    print(f"                                  effective history dimension d=3p={3*p:.4f}")

    omega_m, scale_two, chi2_two = fit_two_channel(rows, y, covariance)
    h0_two = C_KM_S / (scale_two * R_D_MPC)
    omega_history = 1.0 - omega_m
    total_rate = 3.0 * (h0_two / 3.0856775814913673e19) * (1.0e9 * 365.25 * 86400.0) / np.log(2.0)
    print(
        f"free two-channel model           chi2={chi2_two:8.2f}  dof={len(y)-2:2d}  "
        f"Om={omega_m:.4f}  H0(rd_ref)={h0_two:6.2f}"
    )
    print(
        f"                                  present rates: matter={total_rate*np.sqrt(omega_m):.4f}, "
        f"history={total_rate*np.sqrt(omega_history):.4f} bits/Gyr"
    )

    # A predeclared low/high-redshift split: fit p and scale below z=1, predict above z=1.
    train = np.asarray([row.z < 1.0 for row in rows])
    test = ~train
    train_rows = [row for row, keep in zip(rows, train) if keep]
    test_rows = [row for row, keep in zip(rows, test) if keep]
    p_train, scale_train, chi2_train = fit_power(
        train_rows, y[train], covariance[np.ix_(train, train)]
    )
    test_shape = prediction_shape(test_rows, lambda z: power_law_geometry(z, p_train))
    test_residual = y[test] - scale_train * test_shape
    test_covariance = covariance[np.ix_(test, test)]
    chi2_test = float(test_residual @ np.linalg.solve(test_covariance, test_residual))
    print()
    print("Held-out redshift test (fit z<1, no refit at z>=1)")
    print(f"  train: p={p_train:.4f}, chi2={chi2_train:.2f} for {train.sum()-2} dof")
    print(f"  test:  chi2={chi2_test:.2f} for {test.sum()} points")

    omega_train, scale_two_train, chi2_two_train = fit_two_channel(
        train_rows, y[train], covariance[np.ix_(train, train)]
    )
    two_test_shape = prediction_shape(test_rows, lambda z: lcdm_geometry(z, omega_train))
    two_test_residual = y[test] - scale_two_train * two_test_shape
    chi2_two_test = float(two_test_residual @ np.linalg.solve(test_covariance, two_test_residual))
    print("Held-out two-channel test")
    print(f"  train: Om={omega_train:.4f}, chi2={chi2_two_train:.2f} for {train.sum()-2} dof")
    print(f"  test:  chi2={chi2_two_test:.2f} for {test.sum()} points")


if __name__ == "__main__":
    main()
