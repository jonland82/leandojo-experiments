# Semantic-trajectory atlas of sorting networks

This experiment enumerates every correct four-input comparator network of
length five through eight. A trajectory records the Shannon entropy of the
network's output distribution after each comparison, over the uniform set of
all 24 input permutations. This describes how quickly a program collapses
order uncertainty without referring to comparator names or source style.

Correct trajectories are embedded from their normalized entropy curves. The
atlas distinguishes exact five-comparator optima, genuine longer solutions,
and redundant networks that reach a sorted state before their final operation.

Run from the repository root:

```powershell
python experiments/creativity-sorting-atlas/scripts/analyze.py
```
