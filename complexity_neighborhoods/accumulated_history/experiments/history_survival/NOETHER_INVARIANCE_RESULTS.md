# Noether-shell invariance-gate results

Run on 2026-08-24 on the local laptop. This is the invariance follow-up to
`NOETHER_CARDINALITY_RESULTS.md`.

## Question

The conservation-shell experiment found that subtracting the universal
codimension divergence leaves a convergent finite cell-count term. This test
asks whether that finite term is invariant under grid shape and canonical
coordinates, and whether Liouville phase volume supplies a better object.

## Raw intersecting-cell count

The harmonic-oscillator shell was counted using equal-area rectangular cells
with aspect ratios 0.5, 1, and 2, plus square cells after the canonical shear

```text
Q = x,     P = p + x.
```

At the finest common area scale, the codimension-subtracted finite terms were:

| scheme | finite part (bits) |
|---|---:|
| aspect 0.5 | 0.1773 |
| aspect 1 | 0.4906 |
| aspect 2 | 0.1773 |
| canonical shear | 0.2225 |

The spread is 0.3134 bits. Convergence within each scheme therefore does not
make the finite cell-count term coordinate or grid invariant.

## Liouville band density

The replacement estimator measures the Liouville phase volume in the fixed
physical energy band `|H-E| <= 0.05`, divided by the band width. Cell area,
rather than cell incidence, weights the quadrature.

For the harmonic oscillator, the exact band density is `6.283185`. Across all
grid aspects and the canonical shear, numerical values ranged from `6.283000`
to `6.290000`: a 0.111% cross-scheme spread and at most 0.108% error.

For the anharmonic Hamiltonian

```text
H = p^2/2 + x^2/2 + 0.5 x^4/4,
```

the exact band density is `5.061875`. Numerical values ranged from `5.058000`
to `5.071000`: a 0.257% cross-scheme spread and at most 0.180% error.

## Verdict

The raw cell-count finite part fails the invariance gate. The Liouville density
of states passes it for two Hamiltonians and under both grid deformation and a
canonical coordinate transformation.

The invariant candidate is therefore phase-space measure, not the number of
intersected cells. It is not yet a dimensionless bit count: converting a phase
volume into a number of states still requires a physical action cell, such as
a quantum `h`, or another independently derived normalization. The earlier
time-slicing divergence also remains. These are now precise requirements for a
continuum history-information rate rather than arbitrary estimator choices.
