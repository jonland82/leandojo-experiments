# Noether/action cardinality results

Run on 2026-08-24 on the local laptop. This experiment directly implements the
construction in J. Landers, [Noether, Energy-Conserving Transitions, and
Distance-to-Conservation](https://jonathan-r-landers.s3.us-east-1.amazonaws.com/noether_and_energy_transition_cardinality.html),
then extends its one-step cardinality to accumulated paths.

## Mapping from the note

The note defines a bounded discretized state space, the subset of next states
compatible with an energy shell, and the fraction `N_cons/N_all`. Here the
Hamiltonian is the unit harmonic oscillator,

```text
H(x,p) = (x^2+p^2)/2,     E0 = 1.
```

No energy tolerance is fitted. A phase-space grid cell is admissible exactly
when the range of `H` over that cell intersects `E0`. Thus the effective energy
tolerance follows from the discretization cell itself.

For accumulated histories, the discrete harmonic-oscillator action supplies
the Euler-Lagrange residual

```text
x_next + (-2 + dt^2) x_now + x_previous = 0.
```

A three-cell transition is admitted exactly when interval arithmetic shows
that its cells contain a zero of this residual. Candidate next cells are given
the same uniform measure used in the note's state-counting interpretation.

## Energy-shell result

| cell width | admissible / total | raw bits | codimension-renormalized bits |
|---:|---:|---:|---:|
| 0.2000 | 63 / 400 | 2.6666 | 0.3446 |
| 0.1000 | 120 / 1,600 | 3.7370 | 0.4150 |
| 0.0500 | 224 / 6,400 | 4.8365 | 0.5146 |
| 0.0250 | 448 / 25,600 | 5.8365 | 0.5146 |
| 0.0125 | 911 / 102,400 | 6.8125 | 0.4906 |

The raw information grows by approximately one bit whenever resolution is
doubled. This is the expected codimension-one scaling: shell cells grow like
`epsilon^-1`, while all phase-space cells grow like `epsilon^-2`. Subtracting
the universal `log2(1/epsilon)` codimension term leaves a convergent finite
part near 0.5 bits.

## Accumulated stationary-action histories

At `dt=0.2`, raw bits per accepted update increased from 2.0919 to 5.9069 over
the same resolution range. After subtracting `log2(1/epsilon)` per update, the
finest values converged near `-0.415`.

At fixed duration 2.4 and cell width 0.025, temporal refinement gave:

| dt | accepted updates | total bits | bits / unit time |
|---:|---:|---:|---:|
| 0.4 | 5 | 24.6457 | 10.2690 |
| 0.2 | 11 | 54.0137 | 22.5057 |
| 0.1 | 23 | 113.5199 | 47.3000 |

## Verdict

The Noether/action construction succeeds in replacing an arbitrary geometric
tolerance with physical admissibility. It also verifies the note's central
claim that conservation acts as a dimensional filter on candidate futures.

It does not yet yield an absolute information-production rate. Raw cardinality
depends logarithmically on spatial resolution, while accumulated bits per unit
time diverge under finer time slicing. Only renormalized finite parts converge.

The surviving routes are therefore to derive either:

1. a physical spatial and temporal cutoff that fixes the raw cardinality; or
2. a renormalized information density/rate whose subtraction and normalization
   are fixed by the theory rather than chosen to match the Hubble scale.
