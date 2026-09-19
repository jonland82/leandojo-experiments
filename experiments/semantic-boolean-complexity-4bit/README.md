# Four-input semantic prediction of exact Boolean formula complexity

This experiment scales the three-input result to the complete universe of
65,536 four-input Boolean functions. It computes exact minimum formula sizes
in NAND, NOR, and AND/OR/NOT; derives syntax-free truth-table descriptors; and
evaluates semantic nearest-neighbor prediction under function, input-orbit,
and exact-descriptor-class holdouts.

The papers in `notes/` are intentionally not modified by this experiment.

Run from the repository root:

```powershell
python experiments/semantic-boolean-complexity-4bit/scripts/run.py
```

Outputs are written to `artifacts/` and summarized in `RESULTS.md`.

The synthesis is exact by induction on gate cost. In a minimum formula, every
proper subformula must also be minimum for the function it computes; otherwise
replacing it would shorten the whole formula. The dynamic program therefore
retains only functions first reached at each cost and combines those minimum
layers. Its implementation is checked by reproducing all existing three-input
minima and by verifying NAND/NOR duality over all 65,536 four-input functions.
