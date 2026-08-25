# Multiscale identifiability-gate results

Run on 2026-08-24 on the local laptop with seed `20260824` and 1,000,000
paired histories.

## Protocol

Uniform physical expansion generated the histories over 10 Gyr. The
conditioner was an identity propagator containing no Hubble parameter.
Compatibility radii were fixed in units of the RMS initial shared-past
separation and scanned at 151 logarithmically spaced values from 0.01 to 5.0.

A nontrivial rate was declared identifiable only if it remained within 10%
across at least one decade of radius, away from zero-survivor and
all-survivor boundaries. A radius calibrated at the reference expansion rate
also had to transfer to other expansion histories within 10%.

## Results

No nonzero expansion history exhibited a one-decade stability plateau.
Moreover, each history could be made to return approximately 0.25 bits/Gyr by
choosing a different radius:

| expansion rate | radius giving about 0.25 bits/Gyr |
|---:|---:|
| `0.5 H_ref` | 0.2429 |
| `1.0 H_ref` | 0.5799 |
| `1.5 H_ref` | 1.0795 |

Calibrating `r=0.5799` at the reference history did not transfer:

| expansion rate | measured rate | required linear rate |
|---:|---:|---:|
| `0.5 H_ref` | 0.0237 | 0.1250 bits/Gyr |
| `1.0 H_ref` | 0.2537 | 0.2500 bits/Gyr |
| `1.5 H_ref` | 0.4921 | 0.3750 bits/Gyr |

Normalization across three initial length scales collapsed exactly, and
changing the time slicing from 20 to 40 to 80 steps changed cumulative distance
by only `1.49e-14`. The failure is therefore not a unit or timestep artifact.

## Verdict

The identifiability gate failed. For continuous physical separations in this
ensemble, the inferred bit rate is materially selected by the compatibility
radius. A single calibrated radius also does not produce the proposed linear
relationship between information rate and Hubble rate across histories.

This rules out proceeding directly to an N-body calculation with the same raw
fixed-radius physical-separation estimator. It does not rule out accumulated
compatible histories generally. The next construction needs a physical
admissibility criterion that fixes coarse graining independently—for example,
an energy/action or conservation-shell tolerance derived from measurement or
discretization uncertainty—before cardinality is converted into bits.
