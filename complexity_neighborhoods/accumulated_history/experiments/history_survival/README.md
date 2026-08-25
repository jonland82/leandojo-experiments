# Compatible-history survival experiment

This is the next active experiment. `run_history_survival.py` implements its
first controlled estimator-validation stage; it does not yet fit cosmology.
The exact-conditioning laptop results are in `RESULTS.md`; the follow-up
misspecification results are in `CONDITIONING_STRESS_RESULTS.md`; and the
full-radius gate is recorded in `MULTISCALE_IDENTIFIABILITY_RESULTS.md`. The
Noether energy-shell and stationary-action extension is recorded in
`NOETHER_CARDINALITY_RESULTS.md`; its coordinate and grid invariance test is in
`NOETHER_INVARIANCE_RESULTS.md`.

## Scientific question

For histories that share a coarse-grained past, does the independently
measured rate at which they become distinguishable have a stable relationship
to cosmological expansion?

## Required construction

1. Define an ensemble of histories with a common coarse-grained initial past
   and controlled perturbations. Do not define the ensemble using the target
   expansion rate.
2. Choose a cumulative, nonnegative path dissimilarity and a fixed tolerance
   `r`. The theorem applies to this fixed-radius construction, not fixed-`k`
   neighbor turnover.
3. Estimate the survival fraction
   `rho(t; r) = P(delta_t <= r | shared past)` and define
   `C(t; r) = -log2 rho(t; r)`.
4. Condition out distinctions predictable from the stipulated reversible
   dynamics. Report the conditional increment contributed by genuinely new
   coarse-grained information.
5. Keep physical, comoving, and abstract history coordinates explicit and do
   not mix their volume changes in one estimator.
6. Predeclare one conversion between the independently measured `dC/dt` and
   expansion, calibrate it on one time/redshift interval, and predict held-out
   intervals.

## Minimum controls

- tolerance and coarse-graining sensitivity;
- fixed-radius versus fixed-cardinality comparison;
- reversible or time-shuffled null dynamics;
- matched ensembles with and without gravitational mode coupling;
- convergence in ensemble size and temporal resolution;
- explicit accounting of any assumed cosmological background.

## Falsification criterion

After one predeclared calibration, the conditional history-survival rate must
predict held-out expansion history with a redshift-independent mapping. A fit
obtained by defining `C(t)` from `H(t)` is a reconstruction, not a test.

## Controlled validation

The laptop-scale validation checks four cases before cosmological histories are
introduced:

- predictable Hubble flow;
- deterministic chaotic divergence;
- independent innovation bits with a known rate;
- deterministic hidden-state revelations under coarse versus complete
  conditioning.

It also checks ensemble convergence, fixed-radius tolerance dependence, and
regrouping two primitive updates into one reporting interval.

Run from the repository root:

```powershell
python complexity_neighborhoods/accumulated_history/experiments/history_survival/run_history_survival.py
python complexity_neighborhoods/accumulated_history/experiments/history_survival/run_conditioning_stress.py
python complexity_neighborhoods/accumulated_history/experiments/history_survival/run_multiscale_identifiability.py
python complexity_neighborhoods/accumulated_history/experiments/history_survival/run_noether_cardinality.py
python complexity_neighborhoods/accumulated_history/experiments/history_survival/run_noether_invariance.py
```
