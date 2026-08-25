# Expansion-history baseline

**Question:** Can a constant number of effective distinctions per doubling of
cosmic age reproduce the DESI DR2 BAO distance-redshift shape?

**Objects and ensemble:** cosmological expansion histories compared through the
published 13-element DESI DR2 BAO vector and covariance.

**Information model:** `C(t) = d_eff log2(t/t*)`, implying a constant power-law
scale factor. This is a stipulated model, not a measured history complexity.

**Coordinates and neighborhoods:** abstract history coordinates; no spatial
neighborhood estimator is used.

**Calibration:** `c/(H0 r_d)` is profiled out. The test concerns redshift shape,
not an absolute prediction of `H0`.

**Falsification criterion and result:** failure of the fixed law to fit or
predict held-out redshifts. All constant-power versions fail strongly. This
rules out constant `d_eff`, not a dynamical conditional-history rate.

Run from the repository root:

```powershell
python complexity_neighborhoods/accumulated_history/experiments/expansion_history/test_history_models.py
```
