# Do similar Lean statements have similar recorded proofs?

## Summary

This experiment tests local cross-view geometry on the same 10,000 theorem--proof pairs used in the semantic-embedding note. It does not call AWS and does not refit the embeddings. For each theorem, it asks whether nearby statements have proofs that are closer than an appropriate random baseline in proof-embedding space.

The answer is **yes, with a clear but incomplete association**. At `k=10`, proofs attached to a theorem's ten nearest statement neighbors have mean cosine similarity **0.6727**, versus **0.6030** for random neighbors matched as same source file, same coarse module but different file, or different coarse module. The difference is **0.0696** (query-theorem bootstrap 95% interval **[0.0687, 0.0705]**).

The top-10 statement and proof neighborhoods overlap by **0.1343** on average, compared with random expectation **0.0010**; **65.6%** of theorems have at least one neighbor in both top-10 lists. Similar statements therefore make similar recorded proofs appreciably more likely, but do not determine them.

This is a local result, not a claim that statements determine proofs. Each theorem still contributes only one recorded proof, and nearby statements can admit very different valid arguments.

## Design

Let `S_i(k)` and `P_i(k)` be theorem `i`'s exact top-`k` neighbors in statement and proof cosine geometry. For each `k`, the analysis reports:

- the mean proof cosine along statement-neighbor pairs;
- a random baseline that preserves whether each pair is from the same source file, the same coarse module but different files, or different coarse modules;
- the difference between those quantities, with a theorem-level bootstrap interval; and
- `|S_i(k) intersect P_i(k)| / k`, the fraction of neighbors shared by the two views.

Exact top-100 neighborhoods are computed from the full 1,024-dimensional matrices. A separate fixed sample of theorem pairs measures global rank correlation and supports a descriptive regression controlling for coarse module, exact source file, and proof-length difference.

## Local neighborhood results

| k | Proof cosine: statement neighbors | Context-matched random | Difference | 95% interval | Neighborhood overlap | Chance overlap | Overlap / chance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.6883 | 0.6136 | 0.0747 | [0.0736, 0.0759] | 0.1466 | 0.0005 | 293.2x |
| 10 | 0.6727 | 0.6030 | 0.0696 | [0.0687, 0.0705] | 0.1343 | 0.0010 | 134.3x |
| 25 | 0.6557 | 0.5922 | 0.0635 | [0.0629, 0.0642] | 0.1254 | 0.0025 | 50.2x |
| 50 | 0.6448 | 0.5862 | 0.0586 | [0.0580, 0.0591] | 0.1263 | 0.0050 | 25.3x |
| 100 | 0.6351 | 0.5822 | 0.0530 | [0.0525, 0.0535] | 0.1347 | 0.0100 | 13.5x |

The overlap statistic is deliberately strict: it requires the same theorem to appear among the top `k` neighbors in both spaces. Its random expectation is `k/(n-1)`.

### Pair-composition check at k=10

Among top-10 statement-neighbor pairs, **54.0%** share a coarse module and **22.1%** share an exact source file; the random baseline matches those categories pair by pair. Exact statement text occurs in only **0.014%** of pairs. Exact recorded proof text occurs in **0.83%**, versus **0.33%** in the matched baseline, so repeated scripts contribute to the effect but cannot by themselves explain it.

The bootstrap interval treats the query theorem as the resampling unit. Because theorems can also recur as neighbors, the interval is descriptive rather than a fully independent-pair confidence interval.

## Global pairwise association and controls

Across 500,000 fixed random theorem pairs, statement and proof cosine similarity have Pearson correlation **0.3477** and Spearman rank correlation **0.3415**. Pair observations are not independent, so these are effect sizes rather than conventional independent-sample significance tests.

In the descriptive standardized regression, the coefficient of statement cosine is **0.3047** after controlling for shared coarse module, shared source file, and absolute log tactic-count difference. Adding statement cosine increases `R^2` by **0.0892**.

### Proof similarity across statement-similarity deciles

| Statement-similarity decile | Mean statement cosine | Mean proof cosine | Pairs |
|---:|---:|---:|---:|
| 1 | 0.4427 | 0.5347 | 50,000 |
| 2 | 0.4799 | 0.5495 | 49,999 |
| 3 | 0.4995 | 0.5576 | 50,001 |
| 4 | 0.5154 | 0.5635 | 50,000 |
| 5 | 0.5298 | 0.5693 | 50,000 |
| 6 | 0.5440 | 0.5756 | 50,000 |
| 7 | 0.5590 | 0.5811 | 50,000 |
| 8 | 0.5761 | 0.5874 | 50,000 |
| 9 | 0.5986 | 0.5962 | 50,000 |
| 10 | 0.6478 | 0.6165 | 50,000 |

## Discussion

The cluster result and the neighborhood result answer different questions. Low statement--proof cluster AMI says the two views do not induce the same global taxonomy. The positive local effect here says that unusually similar statements nevertheless carry some information about how their recorded proofs look along the proof view's own axes. Global partition mismatch and local predictability can therefore coexist.

The context-matched baseline makes the result harder to explain solely by nearby statements sharing a branch of Mathlib or an exact source file. The controlled regression asks the same question across a broad random-pair sample. Neither control removes all leakage: theorem names, reused identifiers, repeated proof scripts, local author conventions, and automation may still contribute.

Most importantly, this is observational and one-proof-per-theorem. It does not show that a similar theorem must have a similar proof, that the observed proof is canonical, or that similarity will improve proof generation. It supplies the empirical premise for the paid retrieval experiment documented in `AWS_PROOF_GENERATION_PLAN.md`.

## Reproduction and provenance

- Records: **10,000**
- Dimensions per view: **1,024**
- Pair sample seed: **0**
- Random theorem pairs: **500,000**
- Runtime: **11.5 seconds**
- Statement embedding SHA-256: `930e01dc5d22debc9824eaf1043c2107c3b19f2243d3b1b7a459b35829d780d4`
- Proof embedding SHA-256: `c58b01459d4fa178b984be05d10fdf83dc3ce1244303370ffaaebaf0d87037cd`
- Manifest SHA-256: `7e9567320ad92d5757ee8c957225749b9cea05ccba5a9405b4e47d27e4694b00`

Run from the repository root:

```powershell
python experiments/semantic-neighborhood-transfer-10000/scripts/analyze_geometry.py
```

The saved top-100 neighbor arrays can be reused as the retrieval candidates for the future generation experiment.
