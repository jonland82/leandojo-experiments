# Validation of the controlled creativity atlases

This experiment performs three predeclared stress tests:

1. Boolean complexity prediction holds out complete input-permutation orbits
   and, separately, complete invariant-descriptor classes.
2. A stratified permutation null preserves output bias, algebraic degree, and
   symmetry while breaking the remaining feature--complexity association.
3. Sorting trajectories replace scalar entropy with the complete distribution
   over all 24 output permutations and quotient global wire relabelings.

Run from the repository root:

```powershell
python experiments/creativity-atlas-validation/scripts/analyze.py
```

The script consumes the frozen Boolean atlas but reconstructs and exhaustively
checks the sorting space independently.
