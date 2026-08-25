# Imperfect-conditioning stress-test results

Run on 2026-08-24 on the local laptop with seed `20260824`.

## Purpose

The exact-conditioning experiment showed that the estimator behaves correctly
when the propagator is known perfectly. This stress test deliberately supplied
an incorrect Hubble rate, an incorrect deterministic force parameter, or a
quantized current state and measured the resulting false conditional novelty.

## Hubble-rate misspecification

The true rate was `1/13.8 Gyr`, trajectories were evolved for 10 Gyr, and the
distance was normalized by the RMS initial pair separation. At the central
fixed cumulative radius `r=0.005`:

| Hubble-rate error | false rate (bits/Gyr) |
|---:|---:|
| 0.0% | 0.0000 |
| 0.1% | 0.0000 |
| 0.5% | 0.0880 |
| 1.0% | 0.3117 |
| 2.0% | 0.5928 |
| 5.0% | 0.9871 |

The false rate first exceeded the cosmological target of 0.25 bits/Gyr at:

| fixed radius | first tested crossing |
|---:|---:|
| 0.0025 | 0.5% H error |
| 0.0050 | 1.0% H error |
| 0.0100 | 2.0% H error |

## Deterministic force misspecification

For the chaotic standard map at fixed radius `r=0.1`, force-parameter errors of
0.5%, 1%, 2%, and 5% produced false rates of 0.0001, 0.0014, 0.0033, and
0.0057 bits/update. The exact model produced zero.

## Coarse-state conditioning

With the correct force law at fixed radius `r=0.1`, state quantization widths of
0.01 and 0.05 radians produced false rates of 0.0085 and 0.0223 bits/update.
Widths at or below 0.001 radians produced no detectable leakage at this radius.

## Verdict

All six behavioral checks passed: exact conditioning gave zero leakage and
leakage increased monotonically with parameter error or coarseness. The main
scientific result is nevertheless cautionary. A percent-level error in an
expansion model can imitate the entire target history-information rate, and
the crossing depends materially on the compatibility radius.

Therefore a cosmological test must not condition on a fitted value of the same
Hubble rate it intends to predict. Its tolerance must also be fixed by an
independent measurement or coarse-graining rule. Until those two requirements
are met, an apparent match near 0.25 bits/Gyr is not distinguishable from
conditioning leakage.
