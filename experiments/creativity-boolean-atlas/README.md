# Semantic atlas of three-input Boolean functions

This experiment places all 256 three-input Boolean functions in a common
semantic coordinate system and overlays exact synthesis measurements from
three formula languages: NAND, NOR, and AND/OR/NOT.

The coordinates use only truth-table-derived invariants: output bias, sorted
variable influences, Walsh--Fourier energy by degree, algebraic degree, and
input-permutation symmetry. Formula syntax is not used to construct the map.
For each language, semantic dynamic programming computes exact minimum gate
count and boundary surprisal

\[
R(f)=-\log_2 P(e\text{ computes }f\mid |e|=L_f^*).
\]

Run from the repository root:

```powershell
python experiments/creativity-boolean-atlas/scripts/analyze.py
```
