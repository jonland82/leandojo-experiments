# Controlled tests of the creativity triad

These experiments turn the proposed creativity heuristic into two finite,
exactly auditable tests:

\[
K(x)=
\underbrace{S(x)}_{\text{structure preserved}}
\underbrace{G(x)}_{\text{compression gain}}
\underbrace{R(x)}_{\text{low prior probability}}.
\]

The multiplication is a logical reminder: none of the three terms is intended
to substitute for another.

## Four-input sorting networks

The representation class is a fixed-length sequence over the six unordered
comparators. Correctness is exhaustive over all 24 permutations, and the prior
is uniform conditional on length.

- No sequence of length at most four sorts all inputs.
- Exactly 12 of 7,776 length-five sequences work.
- Boundary probability: 0.00154321; surprisal: 9.340 bits.
- A verified insertion-network baseline uses six comparators.
- Appending arbitrary comparators preserves correctness while eliminating the
  one-comparison compression gain.

This is a direct finite confirmation of the sorting paper's claim: correct
representations are thinnest at the compression boundary, while correct but
redundant representations proliferate away from it.

## Three-variable NAND synthesis

The representation class is ordered binary NAND formulas with leaves
`x0`, `x1`, and `x2`. Semantic dynamic programming exactly counts formulas by
truth table through 14 gates and synthesizes all 256 Boolean functions. Counts
are independently checked against
\(\mathrm{Catalan}(L)3^{L+1}\).

Three-bit parity is the sharpest target:

- exact minimum: 14 NAND gates;
- it is the only function whose minimum is 14;
- 373,248 of 38,375,290,837,080 size-14 formulas compute it;
- boundary probability: \(9.726\times10^{-9}\);
- boundary surprisal: 26.615 bits;
- compression: 225 gates relative to the declared balanced canonical
  truth-table compiler.

Across all functions, minimum size and boundary surprisal are strongly related
(Spearman \(\rho=0.948\)). In contrast, compiler-relative compression gain and
surprisal are essentially unrelated (\(\rho=-0.037\)). Rarity therefore does
not automatically imply compression, even in this tightly controlled space.

## What is verified

For the reported witnesses, the experiments mechanically establish:

1. semantic correctness over the complete finite input domain;
2. minimum size within the declared language;
3. compression relative to a frozen explicit baseline; and
4. exact surprisal under a frozen explicit syntax prior.

This verifies an **operational instance** of the triad. It does not prove that
the score is representation-independent, that the chosen prior models human
search, or that a human or machine discovery process was creative. The main
scientific gain is that every dependency of the claim is now explicit and
falsifiable.

## Semantic-atlas extension

Two follow-up atlases test which patterns survive a change of representation.
For all 256 Boolean functions, permutation-invariant truth-table and Fourier
descriptors predict exact minimum formula size with leave-one-function-out
\(R^2=0.831\) for NAND, \(0.833\) for NOR, and \(0.920\) for AND/OR/NOT.
There is therefore a substantial semantic component to synthesis difficulty.
It is not fully intrinsic: NAND--NOR rank correlations are \(0.548\) for
minimum size, \(0.376\) for boundary surprisal, and \(0.264\) for compression
gain.

For sorting, the 12 optimal networks all induce the same entropy-loss
trajectory over the 24 inputs. Correct networks fan out to 27, 147, and 497
trajectory types at lengths six, seven, and eight. The fraction that reaches a
sorted state before its final comparator grows from 8.0% to 32.1% to 49.4%.
At entropy resolution the compression boundary appears rigid, while redundant
correct representations proliferate away from it.

Together these results support a qualified claim: semantic structure predicts
where difficult objects live, but the numerical values of rarity and
compression remain dependent on the representation language and prior.

## Validation update

The initial Boolean prediction partly benefited from leaving permuted or
descriptor-identical semantic twins in training. Holding out complete input
orbits reduces \(R^2\) to 0.726--0.859; holding out all 36 invariant descriptor
classes reduces it further to 0.613--0.656. The signal nevertheless exceeds
1,000 null permutations matched on bias, algebraic degree, and symmetry in all
three languages (finite-simulation \(p<0.001\)).

The sorting conclusion is resolution-sensitive. All 12 optima share one
entropy curve, but they induce 12 distinct full output-distribution paths.
The corresponding full-path counts at lengths six through eight are 756,
4,356, and 15,036, compared with only 27, 147, and 497 entropy profiles. Thus
the boundary remains small and rare, but its apparent one-path rigidity was a
coarse-measure artifact.

## Reproduction

```powershell
python experiments/creativity-sorting-networks/scripts/analyze.py
python experiments/creativity-boolean-synthesis/scripts/analyze.py
python experiments/creativity-boolean-atlas/scripts/analyze.py
python experiments/creativity-sorting-atlas/scripts/analyze.py
python experiments/creativity-atlas-validation/scripts/analyze.py
```

Detailed tables, witnesses, figures, and machine-readable results live in the
experiment directories.
