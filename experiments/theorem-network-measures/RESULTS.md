# Results: theorem network measures

The three scores are related but not interchangeable. Across the same 10,000
theorem--proof records used in the semantic experiments, structural complexity
tracks dependency reliance strongly, while semantic connectedness tends to be
lower for more dependent and involved proofs. Statement and proof connectedness
share only moderate rank agreement.

## Dependency reliance

The pinned Lean environment yielded 888,992 exported declarations and
13,264,773 direct kernel-dependency edges. All target ancestor searches
terminated within the frozen 64-level allowance; the greatest observed depth
was 52.

For the primary score $\log(1+D_{0.5})$, the median is **4.668**, with a 5th--95th
percentile range of **2.993--6.123**. A target has a median **39** direct kernel
dependencies and dependency depth **18**. The discount is not driving the
ordering: compared with $\alpha=0.5$, rank correlation remains **0.904** even at
$\alpha=1$ and is higher for the other tested values.

## Semantic connectedness

The 99th-percentile random-pair cosine thresholds are **0.690** for statement
embeddings and **0.755** for proof embeddings. Among exact top-100 neighbors
above those cutoffs, median effective neighborhood size is **31.3** for
statements but only **8.6** for proofs. At least 5% of proofs have no neighbor
above the proof threshold, whereas the statement 5th percentile is 2.7 effective
neighbors.

Statement and proof effective sizes have Spearman correlation **0.490**. Thus a
theorem embedded in a broad statement community is somewhat likely--but far
from guaranteed--to have a proof embedded in a broad proof community. Rankings
remain stable over null-quantile thresholds 0.975--0.995: the lowest tested
correlation with the primary ranking is **0.926**.

## Structural complexity

The equal-weight standardized score combines tactic count, direct kernel
dependency count, dependency depth, and tactic-head entropy. Its median is
**-0.128**, and its 5th--95th percentile range is **-1.030--1.460**. Leaving out
any one component preserves the primary ordering with rank correlation at least
**0.941**.

Complexity correlates **0.734** with dependency mass. This is partly mechanical:
direct dependency count and depth occur in the complexity definition. More
interestingly, complexity correlates **-0.398** with statement connectedness and
**-0.584** with proof connectedness. In this sample, involved proofs tend to sit
in sparser semantic neighborhoods under the frozen embedding rule.

## Interpretation

The scores answer different descriptive questions:

- dependency mass measures how much kernel-visible infrastructure lies behind a
  theorem;
- effective neighborhood size measures local density in the chosen embedding;
- structural complexity measures involvement of the recorded tactic proof, not
  intrinsic mathematical difficulty.

KDE curves in the report summarize empirical distributions; no parametric
distribution is fitted. Semantic neighborhood size is capped at 100, and all
semantic conclusions remain model- and corpus-dependent. The complete figures
and representative tail examples are available in [`index.html`](index.html).
