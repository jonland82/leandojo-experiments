"""Numerical calibration of a neighborhood-thinning cosmology toy model.

This is deliberately dependency-free.  It distinguishes quantities predicted by
the toy ansatz from quantities reconstructed from measured cosmological inputs.
"""

from __future__ import annotations

import math


SECONDS_PER_GYR = 1.0e9 * 365.25 * 24.0 * 60.0 * 60.0
KM_PER_MPC = 3.0856775814913673e19
G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
M_PER_MPC = 3.0856775814913673e22

# Frozen published inputs.  These are not fitted by this script.
AGE_GYR = 13.80  # Planck 2018: approximately 13.8 Gyr
PLANCK_H0 = 67.4  # km s^-1 Mpc^-1
DESI_DR2_H0 = 68.17  # DESI DR2 + CMB, flat LambdaCDM
DESI_DR2_OMEGA_M = 0.3027
SHOES_H0 = 73.04  # 2022 Cepheid-SN Ia distance ladder


def h_km_s_mpc_to_s(hubble: float) -> float:
    return hubble / KM_PER_MPC


def h_s_to_km_s_mpc(hubble: float) -> float:
    return hubble * KM_PER_MPC


def turnaround_radius_mpc(mass_solar: float, h0: float, omega_m: float) -> float:
    """Maximum LambdaCDM turnaround radius using Lambda=3 Omega_L H0^2/c^2."""
    omega_lambda = 1.0 - omega_m
    h_si = h_km_s_mpc_to_s(h0)
    lambda_si = 3.0 * omega_lambda * h_si**2 / C_SI**2
    radius_m = (3.0 * G_SI * mass_solar * M_SUN_KG / (lambda_si * C_SI**2)) ** (1.0 / 3.0)
    return radius_m / M_PER_MPC


def main() -> None:
    age_seconds = AGE_GYR * SECONDS_PER_GYR

    # Parameter-free scale estimate under N(t)=3 log_2(t/t_*), hence a(t)=t/t_*.
    h_age_si = 1.0 / age_seconds
    h_age = h_s_to_km_s_mpc(h_age_si)

    print("Neighborhood-thinning cosmology: numerical calibration")
    print(f"Age input:                         {AGE_GYR:.2f} Gyr")
    print(f"Age-scale prediction H=1/t0:      {h_age:.2f} km s^-1 Mpc^-1")
    print()
    print("Comparison (p_eff = H0 t0; p_eff=1 is the age-scale prediction)")
    for label, measured in (
        ("Planck 2018", PLANCK_H0),
        ("DESI DR2 + CMB", DESI_DR2_H0),
        ("SH0ES 2022", SHOES_H0),
    ):
        p_eff = h_km_s_mpc_to_s(measured) * age_seconds
        fractional_error = (h_age - measured) / measured
        print(
            f"  {label:18s} H0={measured:5.2f}, "
            f"p_eff={p_eff:.4f}, age-model offset={fractional_error:+.2%}"
        )

    # If rho ~ 2^-N and a = rho^-1/3, H=(ln 2/3) dN/dt.
    bit_rate_per_gyr = 3.0 / (math.log(2.0) * AGE_GYR)
    print()
    print("Information-rate dictionary for the H=1/t0 model")
    print(f"  effective distinguishing rate:  {bit_rate_per_gyr:.4f} bits/Gyr")
    print(f"  one effective bit every:         {1.0 / bit_rate_per_gyr:.3f} Gyr")
    print("  equivalently: three bits per doubling of cosmic age")

    # A constant late-time bit rate is exactly Lambda-like in this dictionary.
    omega_lambda = 1.0 - DESI_DR2_OMEGA_M
    h0_si = h_km_s_mpc_to_s(DESI_DR2_H0)
    h_lambda_si = h0_si * math.sqrt(omega_lambda)
    lambda_bit_rate_gyr = 3.0 * h_lambda_si * SECONDS_PER_GYR / math.log(2.0)
    lambda_eff_m2 = 3.0 * h_lambda_si**2 / C_SI**2
    q0 = 0.5 * DESI_DR2_OMEGA_M - omega_lambda
    print()
    print("Late-time Lambda dictionary (reconstruction, not a prediction)")
    print(f"  Omega_Lambda:                    {omega_lambda:.4f}")
    print(f"  H_Lambda:                        {h_s_to_km_s_mpc(h_lambda_si):.2f} km s^-1 Mpc^-1")
    print(f"  required constant bit rate:      {lambda_bit_rate_gyr:.4f} bits/Gyr")
    print(f"  Lambda_eff:                      {lambda_eff_m2:.3e} m^-2")
    print(f"  LambdaCDM q0:                    {q0:.3f} (coasting toy model gives 0)")

    print()
    print("Standard-gravity turnaround scale expressed in the same rate language")
    for mass in (1.0e12, 2.0e12, 1.0e15):
        radius = turnaround_radius_mpc(mass, DESI_DR2_H0, DESI_DR2_OMEGA_M)
        print(f"  M={mass:.1e} solar masses:        R_TA,max={radius:.3f} Mpc")


if __name__ == "__main__":
    main()
