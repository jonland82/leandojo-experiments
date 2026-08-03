# Proof style and mathematical domain at 10,000 proofs

## Summary

The larger experiment supports the original result. Proof style and mathematical domain have a
detectable but weak association. On 8,882 proofs with usable assignments in both views, adjusted
mutual information is 0.0153 and Cramer's V is 0.102. The corresponding values in the original
1,940-proof experiment were 0.0157 and 0.113.

That agreement is more informative than the nominal significance test. With 10,000 observations,
even a small departure from independence is easy to detect. The effect size says that the two
topic systems do not closely align: domain is not a disguised proof-style label, and proof style
is not a disguised subject label.

The result also preserves the important asymmetry seen in the smaller experiment. Domain topics
align substantially with the Lean source hierarchy (module AMI 0.241), while style topics barely
do (module AMI 0.025). Mathematical subject is related to where a theorem lives in mathlib; proof
procedure travels much more freely across that structure.

## Experimental design

The benchmark snapshot contains 52,187 theorem records with nonempty tactic traces in the random
split. The `aws-10000` profile selects 10,000 of them uniformly without replacement using random
seed 0. It draws from train, validation, and test because this is an unsupervised descriptive
experiment, not an evaluation that trains on one split and reports predictive accuracy on another.

| Source split | Available tactic proofs | Selected proofs |
|---|---:|---:|
| Train | 50,247 | 9,637 |
| Validation | 979 | 185 |
| Test | 961 | 178 |
| **Total** | **52,187** | **10,000** |

The mean proof length is 4.19 traced tactics, the median is 2, and the range is 1 to 123. The
original sample had mean 3.92 and the same median of 2.

The two representations remain deliberately separate:

1. **Style:** tactic-head unigrams and adjacent tactic-head bigrams.
2. **Domain:** explicitly annotated premise names and their top-level namespaces.

Each view is sublinear TF-IDF weighted and independently factorized by nonnegative matrix
factorization. Candidate resolutions are 4, 6, 8, 10, 12, 14, and 16 topics. For each candidate,
one full-data reference fit is compared with four fits on independent 80% subsamples after optimal
one-to-one topic alignment. The reconstruction-curve elbow among stable candidates chooses the
reported exploratory resolution.

The full matrices are:

| View | Shape | Nonzero entries | Proofs with signal | Selected topics |
|---|---:|---:|---:|---:|
| Style | 10,000 x 1,760 | 56,788 | 9,998 | 8 |
| Domain | 10,000 x 7,521 | 52,111 | 8,882 | 10 |

The two proofs without a style assignment have zero loading after factorization. The 1,118 proofs
without a domain assignment contain no premise feature retained by the domain vectorizer. They
remain in the style analysis but are excluded from cross-view comparisons.

## Topic structure

The style model chooses eight topics rather than the ten selected in the smaller sample. This is a
resolution change, not a failure of stability: all candidate style models through 14 topics have
mean subsample stability above 0.999. The elbow moves because the larger vocabulary and sample
produce a smoother reconstruction curve. No selected value of K should be read as the true number
of proof styles.

| Style topic | Short label | Dominant assignments | Mean tactics |
|---:|---|---:|---:|
| 0 | `rw` | 1,574 | 1.41 |
| 1 | `simp` | 2,023 | 1.85 |
| 2 | `exact` | 1,099 | 3.38 |
| 3 | `simpa` | 632 | 2.38 |
| 4 | `simp_rw` | 457 | 3.45 |
| 5 | `ext` | 529 | 4.19 |
| 6 | `have` / `apply` / `refine'` | 2,930 | 8.26 |
| 7 | `rfl` / `cases` | 754 | 3.66 |

The main procedural axis is clear. Rewriting and normalization topics contain very short proofs,
while the `have` / `apply` / `refine'` topic contains longer, explicitly structured arguments. The
larger sample also separates `exact`, `simpa`, `simp_rw`, and extensionality more cleanly.

The domain model again chooses ten topics. Its labels are broader because namespace tokens absorb
families of related premises.

| Domain topic | Representative content | Assignments | Mean tactics |
|---:|---|---:|---:|
| 0 | Category theory and algebraic geometry | 541 | 5.17 |
| 1 | Sets, images, intersections, preimages | 1,154 | 4.46 |
| 2 | Lists and arrays | 516 | 4.67 |
| 3 | Finsets, multisets, matrices, functions | 1,032 | 4.95 |
| 4 | Measure and probability theory | 777 | 5.04 |
| 5 | Natural numbers, finite types, commutative arithmetic | 1,160 | 4.46 |
| 6 | Polynomials, multisets, and ring homomorphisms | 960 | 4.26 |
| 7 | Integers and rationals | 439 | 4.23 |
| 8 | Filters, convergence, and metric structure | 766 | 4.83 |
| 9 | Real and complex arithmetic | 1,537 | 3.83 |

These are useful empirical bundles, not claims about a canonical taxonomy of mathematics. Some
topics are recognizable branches, while others combine namespaces that share formal machinery or
frequently co-occur in explicit tactic references.

## Cross-view alignment

For the 8,882 proofs assigned in both views, let the contingency table cross ten domain topics with
eight style topics. The projection-free comparison gives:

| Statistic | 1,940 proofs | 10,000 proofs |
|---|---:|---:|
| Proofs assigned in both views | 1,628 | 8,882 |
| Adjusted mutual information | 0.0157 | 0.0153 |
| Normalized mutual information | 0.0273 | 0.0170 |
| Pearson chi-squared | 188.44 | 642.14 |
| Cramer's V | 0.113 | 0.102 |
| Largest within-domain style share | 30.5% | 36.6% |
| Median of the ten within-domain maxima | 23.1% | 31.1% |

None of 20,000 fixed-margin permutations reached the observed chi-squared statistic, giving a
Monte Carlo upper bound below 5e-5. This establishes a departure from independence, but it does
not establish strong alignment. Cramer's V remains small and AMI is almost unchanged from the
original sample.

The row maxima are somewhat larger in the new table partly because the style model now has eight
columns rather than ten and its largest topic contains almost 30% of style-assigned proofs. They
should not be treated as an effect-size comparison in isolation. AMI and Cramer's V are the more
appropriate summaries across the two resolutions.

## Relation to library structure

The larger sample strengthens the distinction between subject and procedure:

| View | Module AMI, 1,940 proofs | Module AMI, 10,000 proofs |
|---|---:|---:|
| Style topics | 0.020 | 0.025 |
| Domain topics | 0.226 | 0.241 |

Domain assignments agree moderately with the first two components of the theorem's mathlib file
path. Style assignments do not. This is expected for the domain representation because premises
and namespaces are embedded in the organization of the library, but the magnitude difference is
still useful evidence: the two models are recovering genuinely different structures.

## Interpretation

The central claim survives a fivefold increase in sample size. Branches of formal mathematics are
connected in more than one way. A dependency or premise graph captures shared objects, definitions,
and results. A proof-style map captures recurring procedures such as rewriting, normalization,
extensionality, case analysis, and intermediate construction. Those geometries intersect, but they
are not interchangeable.

This makes the weak alignment interesting rather than disappointing. A method can be portable
across subjects, and a single subject can support several methods. In a sufficiently large formal
library, procedural kinship may reveal relationships that are muted or absent in the conventional
branch structure of mathematics.

The experiment remains descriptive. Topic labels are induced from selected features; automation
can hide internally used lemmas; dominant assignments discard mixture information; and source
modules are only a rough external label. The strong next test is to compare full topic mixtures and
proof-state transitions, then ask whether a procedure-based network predicts useful cross-domain
transfer in premise or tactic selection.

## AWS execution record

The run used one On-Demand `m7i.2xlarge` instance in `us-east-1` with Amazon Linux 2023 and pinned
versions of NumPy 2.1.3, SciPy 1.15.3, and scikit-learn 1.6.1.

| Measurement | Value |
|---|---:|
| Analysis wall time | 63.96 seconds |
| CPU utilization reported by `time` | 379% |
| Peak resident memory | 2.50 GiB |
| Result payload | 6.23 MB |

The staging bucket, instance profile, role, and instances were deleted after artifact retrieval.
The brief failed setup attempt and successful run together consumed only a few instance-minutes, so
the estimated EC2 and EBS charge is well below $0.10.

## Reproduction and artifacts

From the repository root, with the benchmark extracted under `data/`:

```powershell
python pipeline.py --profile aws-10000
```

The canonical outputs are:

- `artifacts/stats.json`: diagnostics, topics, entropies, module AMI, and layout statistics.
- `artifacts/proofs.json`: per-proof assignments, mixtures, scripts, and coordinates.
- `artifacts/cross_view.json`: the contingency table, alignment statistics, and permutation test.
- `artifacts/run.log`: the AWS console output and resource measurements.

The original experiment remains independently runnable with:

```powershell
python pipeline.py --profile small-1940
```

Its canonical rerun artifacts live in `../small-1940/artifacts/`; the historical root-level
artifacts remain intact for the existing viewer and note.
