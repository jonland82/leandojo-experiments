# CAMELS binding and neighborhood-turnover study

**Question:** Does local spatial-neighborhood turnover accompany expansion or
instead concentrate in gravitationally structured regions?

**Objects and ensemble:** matched CAMELS subhalos across adjacent snapshots.

**Estimator:** turnover of fixed-`k` nearest-neighbor membership and
Jensen-Shannon change of Gaussian spatial-neighborhood weights. This is not a
fixed-radius thinning estimator.

**Coordinates:** neighborhood changes use comoving positions; physical patch
expansion restores the background scale-factor contribution separately.

**Assumed background:** LambdaCDM is built into CAMELS. The experiment cannot
establish whether information causes the background expansion.

**Result:** turnover is strongest in dense or bound regions and is largely
predicted by peculiar velocity. This supports a binding/rearrangement reading
and does not falsify compatible-history survival.

Run from the repository root, for example:

```powershell
python complexity_neighborhoods/spatial_neighborhoods/binding/test_conditional_information.py
python complexity_neighborhoods/spatial_neighborhoods/binding/test_conditional_information.py --realizations 0 1 2 3 4
```
