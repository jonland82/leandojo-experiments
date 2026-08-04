# Tactic Style and Mathematical Domain in Lean Proofs

## An exploratory soft-topic analysis of 1,940 LeanDojo Benchmark 4 proofs

## 1. Question

This experiment asks two related but distinct questions:

1. Which **procedural styles** recur in human-written Lean tactic proofs?
2. Which **mathematical domains** recur in the premises explicitly cited by those proofs?

The distinction matters. Tactic names such as `rw`, `apply`, and `cases` describe how a proof
proceeds. Premise names such as `List.length_append` or
`CategoryTheory.Category.assoc` directly expose its subject matter. Combining the two in one
feature matrix produces clusters that conflate style and domain, so this version models them
separately.

The result is descriptive rather than predictive. It does not claim that Lean proofs have a
true number of discrete kinds. Instead, non-negative matrix factorization gives every proof a
mixture of reusable topics. A dominant topic is retained only for summaries and coloring the
viewer.

## 2. Data

| | |
|---|---|
| Source | LeanDojo Benchmark 4, `random` validation and test splits |
| Included | Every theorem with a nonempty `traced_tactics` list |
| Validation / test | 979 / 961 proofs |
| Total | **1,940 proofs** |
| Mean proof length | **3.92 tactic steps** |
| Median / min / max | **2 / 1 / 85** |

Using both complete held-out splits avoids selecting the first portion of one file. Term-mode
proofs without traced tactics remain outside the scope of the experiment because they provide
no tactic sequence to model.

LeanDojo's annotated tactics identify premises explicitly referenced in tactic syntax. They
should not be read as a complete trace of every lemma used internally by automation such as
`simp`.

## 3. Representations

### 3.1 Style documents

For a theorem with tactic-head sequence

$$
(h_1,h_2,\ldots,h_L),
$$

the style document contains tactic unigrams and adjacent bigrams:

$$
\Phi_{\mathrm{style}} =
\{\texttt{TAC\_}h_i : 1\le i\le L\}
\uplus
\{\texttt{BIGRAM\_}h_i\texttt{\_\_}h_{i+1} : 1\le i<L\}.
$$

The bigrams retain a small, transparent amount of ordering information: `intro → apply` and
`apply → intro` are different features. Premises and raw argument identifiers are excluded,
so local variable names and mathematical namespaces cannot dominate the style view.

After removing features appearing in fewer than two proofs, the style matrix has shape
**1,940 × 639**, with **9,962 nonzeros** and density **0.804%**.

### 3.2 Domain documents

For every premise name $p$ explicitly annotated by LeanDojo, the domain document contains its
full name and, when qualified, its top-level namespace:

$$
\Phi_{\mathrm{domain}} =
\{\texttt{PREM\_}p,\ \texttt{NS\_}\nu(p)\}.
$$

The namespace feature intentionally provides a coarse subject signal while the full name
retains specificity. The matrix has shape **1,940 × 1,385**, with **7,045 nonzeros** and density
**0.262%**. A total of **1,629 proofs** have at least one retained premise feature; the other
311 receive no domain topic.

### 3.3 TF–IDF

Both views use the same weighting. For count $c(w,t)>0$,

$$
\operatorname{tf}(w,t)=1+\log c(w,t), \qquad
\operatorname{idf}(w)=\log\frac{1+N}{1+\operatorname{df}(w)}+1.
$$

Each resulting row is normalized to unit L2 norm. This is exactly the configuration implemented
by scikit-learn's `TfidfVectorizer` with sublinear term frequency, smoothed IDF, and L2
normalization.

## 4. Soft topics with NMF

For each nonnegative TF–IDF matrix $X$, non-negative matrix factorization solves

$$
\min_{W,H\ge0}\frac12\lVert X-WH\rVert_F^2,
$$

where rows of $H$ describe topics over features and rows of $W$ give proof-specific topic
weights. To make weights comparable within a proof, each nonzero row is normalized:

$$
\theta_{ik}=\frac{W_{ik}}{\sum_j W_{ij}}.
$$

These weights are useful mixture coordinates, but NMF is not a probabilistic topic model and
$\theta$ should not be interpreted as a calibrated probability. The dominant topic
$\arg\max_k\theta_{ik}$ is a display convenience, not a hard claim about membership.

Mixture entropy is reported on a normalized 0–1 scale:

$$
E_i=-\frac{\sum_k\theta_{ik}\log\theta_{ik}}{\log K}.
$$

Zero means one topic carries all weight; one means the weights are uniform.

## 5. Choosing a useful resolution

Candidate resolutions are $K\in\{4,6,8,10,12,14,16\}$. Two diagnostics are computed:

- **Relative reconstruction error:** $\lVert X-WH\rVert_F/\lVert X\rVert_F$.
- **Subsample component stability:** fit NMF to four independent 80% subsamples, optimally
  align each fit's topics to the full-data topics with the Hungarian algorithm, and average
  their cosine similarities.

The selected resolution is the maximum-distance elbow of the reconstruction curve among fits
whose stability is at least 0.75. This rule chooses **10 topics** for both views. It is a
repeatable visualization heuristic, not an estimate of a true latent topic count.

### Style diagnostics

| topics | relative reconstruction error | stability | stability SD |
|---:|---:|---:|---:|
| 4 | 0.7731 | 1.000 | 0.000 |
| 6 | 0.7446 | 0.974 | 0.049 |
| 8 | 0.7210 | 0.998 | 0.001 |
| **10** | **0.7013** | **0.998** | **0.000** |
| 12 | 0.6848 | 0.920 | 0.050 |
| 14 | 0.6697 | 0.947 | 0.054 |
| 16 | 0.6563 | 0.915 | 0.037 |

### Domain diagnostics

| topics | relative reconstruction error | stability | stability SD |
|---:|---:|---:|---:|
| 4 | 0.9523 | 0.996 | 0.002 |
| 6 | 0.9342 | 0.981 | 0.016 |
| 8 | 0.9181 | 0.965 | 0.037 |
| **10** | **0.9038** | **0.996** | **0.001** |
| 12 | 0.8926 | 0.998 | 0.000 |
| 14 | 0.8829 | 0.981 | 0.017 |
| 16 | 0.8746 | 0.925 | 0.027 |

High component stability means the main feature bundles recur under subsampling. It does not
by itself prove that they are the only or objectively correct semantic categories.

## 6. Results

### 6.1 Style topics

| # | dominant proofs | mean steps | shorthand | leading features |
|---:|---:|---:|---|---|
| 0 | 379 | 1.7 | `simp` | `simp`, `simp → simp`, `rw → simp` |
| 1 | 342 | 1.8 | `rw` | `rw`, `rw → rw`, `rw → simp` |
| 2 | 209 | 3.7 | `exact` | `rw → exact`, `exact`, `simp → exact` |
| 3 | 136 | 2.0 | `simpa` | `simpa`, `simpa → simpa`, `simpa → simp` |
| 4 | 98 | 3.4 | `ext` | `ext`, `ext → simp`, `ext → rw` |
| 5 | 337 | 8.3 | `refine' / have / rwa` | `refine'`, `have`, `rwa` |
| 6 | 97 | 3.3 | `rfl` | `rfl`, `rw → rfl`, `simp → rfl` |
| 7 | 87 | 4.3 | `simp_rw` | `simp_rw`, `simp_rw → exact`, `exact` |
| 8 | 60 | 3.6 | `cases` | `cases`, `cases → rfl`, `cases → simp` |
| 9 | 193 | 6.7 | `apply` | `apply`, `rw → apply`, `intro` |

The clearest axis is procedural. `simp`, `rw`, `simpa`, and `rfl` dominate short proofs, while
the `refine' / have / rwa` and `apply / intro` topics dominate longer structured proofs. The
mean normalized mixture entropy among the **1,938 proofs with nonzero modeled weight** is
**0.272**: most proofs have a clear leading style topic, while retaining secondary weights
rather than being forced into a single category. Two sparse proofs receive no style topic.

As a check on the intended separation, adjusted mutual information between dominant style
topic and the theorem's coarse file module is only **0.020**. Tactic style therefore carries
almost no coarse module signal in this sample.

### 6.2 Domain topics

| # | dominant proofs | shorthand | characteristic signal |
|---:|---:|---|---|
| 0 | 242 | sets and general equality | `Set.univ`, `rfl`, `Eq.symm` |
| 1 | 93 | category theory | `Category.assoc`, `id_comp`, `comp_id` |
| 2 | 90 | lists | `List.length`, `List.prod_cons`, `List.length_append` |
| 3 | 175 | naturals and finite types | `Nat.mul_comm`, `Nat.cast_one`, `Nat.ModEq` |
| 4 | 156 | measure theory | `Measure.sum_apply`, `L1.integral`, `Measure.restrict_apply` |
| 5 | 161 | polynomial algebra | `algebraMap`, `Polynomial.C_mul_X_pow_eq_monomial` |
| 6 | 266 | finite combinatorics and algebra | `Finset.mem_filter`, `rfl`, `Bool.true` |
| 7 | 165 | real and complex analysis | `mul_comm`, `Real.arccos`, `Real.Angle.induction_on` |
| 8 | 151 | filters and convergence | `Filter.le_principal_iff`, `tendsto_comap_iff`, `Tendsto.comp` |
| 9 | 130 | integers and rationals | `Int.mul_ediv_cancel_left`, `Int.add_zero`, `Int.add_comm` |

The mean normalized entropy among proofs with premise signal is **0.320**. Adjusted mutual
information between dominant domain topic and coarse theorem module is **0.226**: a moderate,
not perfect, association. This is expected because premise namespaces explicitly encode
mathematical vocabulary. The useful result is the degree and shape of the association, not
the surprising discovery of domains from hidden information.

## 7. Visualization

The 3-D coordinates are derived from the **style matrix**, independently of NMF topic weights:

1. 64-component truncated SVD;
2. row normalization;
3. projection to three dimensions with PCA or t-SNE.

The SVD embedding reports 73.4% cumulative explained variance. The three PCA axes explain
17.9%, 10.0%, and 6.6%, totaling **34.4%**. PCA is the best three-dimensional linear
reconstruction in its objective, but it still discards most variation. t-SNE emphasizes local
neighborhoods and should not be read as preserving global distances or blob sizes.

The viewer can color the same style geometry by dominant style or domain topic. This makes the
relationship between procedural proximity and cited mathematical vocabulary visible without
using the colors to construct the layout.

## 8. What can and cannot be concluded

Supported by this experiment:

- Frequent tactic idioms form stable, interpretable nonnegative components.
- Short normalization/rewrite proofs and longer structured proofs occupy different style modes.
- Premise vocabulary yields recognizable mathematical-domain topics with moderate module
  association.
- Style and domain are empirically distinct in this representation: their associations with
  file modules differ sharply.

Not established by this experiment:

- that there are exactly ten proof styles or ten domains;
- that a dominant topic is a discrete proof species;
- that premise annotations include every fact used internally by automation;
- that proof length measures theorem difficulty;
- that the held-out sample represents all of `mathlib4` without sampling uncertainty;
- that separation in t-SNE is statistical evidence of a cluster.

The most important omitted signal remains the goal state before and after each tactic. This
experiment describes human-written tactic vocabulary, not the full semantics of the proof.

## 9. Reproducing

```text
python pipeline.py       # fit both views and regenerate out/ plus app/data.js
python scripts/summarize.py      # compact console summary
python scripts/report_tables.py  # regenerate Markdown diagnostic/topic tables
```

Open `app/index.html` directly in a browser. The viewer has no CDN or server dependency.
Controls include orbit, zoom, pan, theorem/file search, PCA/t-SNE layout, proof-length sizing,
and style/domain coloring.

## 10. Artifacts

| Path | Contents |
|---|---|
| `pipeline.py` | loading, features, NMF diagnostics, topics, layouts, artifacts |
| `out/stats.json` | all reported statistics and topic descriptions |
| `out/proofs.json` | proof coordinates, mixture summaries, dominant topics, scripts |
| `app/data.js` | script-loadable copy of `out/proofs.json` |
| `app/index.html`, `app/viewer.js`, `app/style.css` | dependency-free explorer |
| `scripts/` | cross-view analysis and small artifact/reporting utilities |

Random seed 0 is fixed for NMF reference fits, subsampling, SVD, and t-SNE. Reproducibility is
expected within the recorded software environment; exact floating-point identity across
library versions and platforms is not promised.
