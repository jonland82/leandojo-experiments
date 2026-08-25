# Complexity-neighborhoods research program

This directory is organized by scientific question rather than by whether an
experiment produced a favorable result.

## Claims and status

1. **Proved:** under a cumulative additive distance with nonnegative coordinate
   contributions, fixed-radius neighborhoods can only thin as distinguishing
   coordinates are appended. See `foundation/`.
2. **Hypothesized:** the loss of compatible alternative histories defines a
   relational volume whose scale can be identified with cosmological expansion.
   This physical dictionary is not implied by the theorem.
3. **Ruled out as complete dynamics:** a constant number of effective bits per
   logarithmic interval of cosmic time, including the fixed laws
   `a(t) ~ t` and `a(t) ~ t^(4/3)`.
4. **Inadequate proxy:** total causal-past four-volume, raw or logarithmic, with
   a redshift-independent coupling to expansion.
5. **Active hypothesis:** expansion tracks the conditional rate at which
   alternative histories with a shared past cease to be mutually compatible.
6. **Validated estimator behavior:** exact dynamical conditioning removes
   predictable expansion and deterministic chaos while recovering injected
   unresolved distinctions.
7. **Current invariant candidate:** Noether constraints define dynamical
   admissibility, while Liouville phase volume survives the tested grid and
   canonical-coordinate changes.
8. **Finite constrained-history result:** reduced Liouville measure induces a
   complete-history measure by pushforward through deterministic dynamics.
   Lapse weighting removes clock-density artifacts and the gauge-fixed
   determinant removes orbit-counting artifacts in the parametrized-oscillator
   test. Extension to gravitational constraints remains open.
9. **Separate result:** spatial halo-neighborhood turnover primarily tracks
   peculiar motion and gravitational binding. It is not a direct test of the
   active compatible-history hypothesis.

## Directory map

- `foundation/` — the relational-shape theorem and its original paper.
- `accumulated_history/` — the active compatible-history cosmology program,
  retained baselines, the accumulated-time note, its Noether/Liouville sequel,
  and the invariant classical-history paper.
- `spatial_neighborhoods/` — local spatial-neighborhood and binding studies.
- `references/` — bibliography shared by the papers.
- `archive/no_go_paper/` — completed 2026-08-21 synthesis of the first round of
  cosmological and simulation tests.

## Standard experiment contract

Each active experiment should state:

- its scientific question and falsification criterion;
- the objects and comparison ensemble;
- the complexity or information functional;
- whether neighborhoods use a fixed radius or fixed cardinality;
- whether coordinates are physical, comoving, or abstract history coordinates;
- which dynamics and cosmological background are assumed;
- which quantities are predicted independently and which are calibrated.

The completed estimator sequence and its next-stage requirements are recorded
in `accumulated_history/experiments/history_survival/README.md`; the current
synthesis is `accumulated_history/invariant_history_note/`.
