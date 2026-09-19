# Exact creativity test: four-input sorting networks

This experiment exhaustively aggregates every comparator sequence for four
inputs through length 12. A semantic dynamic program groups sequences inducing
the same map on all \(4!=24\) input permutations, while retaining exact syntax
counts. Thus correctness, minimum comparison count, and conditional rarity are
all exact rather than sampled.

For a length-\(L\) sequence \(A\), the controlled triad is

\[
S(A)=\mathbf 1[A\text{ sorts all 24 inputs}],\quad
G(A)=\max\{0,6-L\},\quad
R_L=-\log_2 \Pr(S=1\mid L),
\]

where six comparisons is the fixed insertion-network baseline and the prior is
uniform over the six possible comparators at each position. The operational
creativity score is \(K(A)=S(A)G(A)R_L\). This score is meaningful only within
the declared representation class, prior, and baseline.

Run from the repository root:

```powershell
python experiments/creativity-sorting-networks/scripts/analyze.py
```

