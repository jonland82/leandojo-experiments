# Theorem network measures

This experiment assigns three kinds of per-theorem magnitude to the
same 10,000 theorem--proof pairs used by `semantic-embeddings-10000`: dependency
reliance, semantic connectedness, and structural proof complexity. They remain
separate because a proof can be complex but isolated, or simple but built on a
large dependency network.

## Data and notation

The targets are a seed-0 uniform sample of 10,000 records with nonempty
`traced_tactics` from the LeanDojo benchmark's random train, validation, and test
splits. Every record supplies `full_name`, `file_path`, source coordinates, the
Mathlib commit, and a sequence of traced tactic steps. Each step contains its
tactic text; the second element of `annotated_tactic` lists explicitly
referenced declarations and their source locations. Statement source is joined
from `corpus.jsonl` using `full_name` and `file_path`.

Let $i$ denote one target theorem. The 10,000 targets define the population on
which scores and standardizations are reported. A dependency graph may include
additional declarations from the same Mathlib commit so that upstream structure
is not artificially cut off at the sample boundary.

## 1. Transitive dependency mass

**Intuition:** How reliant is it on prior mathematics?

Build a directed declaration graph $G=(V,E)$ with an edge $u\to v$ when the
declaration $v$ directly depends on $u$. For the target $i$, let $A(i)$ be all
upstream declarations that can reach $i$, and let $d(u,i)$ be the length of the
shortest directed path from $u$ to $i$. Define

$$
D_\alpha(i)=\sum_{u\in A(i)}\alpha^{d(u,i)}, \qquad 0<\alpha\leq1.
$$

The analysis imports all of Mathlib at the dataset's pinned commit and exports
the constants appearing in each declaration's kernel-checked type and value
(proof term). This recovers dependencies hidden inside tactics such as `simp`.
LeanDojo's `full_name` and `file_path` map each target to its environment name;
source-private declarations are resolved to Lean's `_private` names. The tactic
annotations remain useful metadata, but they do not define this graph.

$\alpha=1$ counts distinct upstream declarations. The frozen primary value is
$\alpha=0.5$, which discounts remote infrastructure. We report
$\log(1+D_{0.5}(i))$, direct dependency count, and maximum shortest-path depth.
The search allows 64 levels; every target's closure terminated by level 52.

**Interpretation:** larger values mean that the theorem rests on a broader or
deeper body of prior declarations.

## 2. Effective semantic neighborhood

**Intuition:** How connected is it semantically?

This method uses the existing statement and proof embeddings separately. Their
text inputs are

```text
Lean 4 theorem statement:
{statement}
```

and

```text
Lean 4 tactic proof:
{newline-joined traced tactic text}
```

Each text was embedded by Cohere Embed v4 (`cohere.embed-v4:0`) through Amazon
Bedrock with input type `clustering`, producing a 1,024-dimensional vector. For
each view, vectors are L2-normalized,

$$
x_i=\frac{e_i}{\lVert e_i\rVert_2},
$$

so the dot product $x_i^\top x_j$ equals cosine similarity. In each view, let
$\mathcal N_{100}(i)$ be the exact 100 nearest neighbors. The threshold $\tau$
is the 99th percentile of 500,000 fixed random-pair cosines in that view. For
$j\in\mathcal N_{100}(i)$, define

$$
w_{ij}=\max\{0,x_i^\top x_j-\tau\}, \qquad
W(i)=\sum_{j\in\mathcal N_{100}(i)}w_{ij}, \qquad
p_{ij}=\frac{w_{ij}}{W(i)}.
$$

For $W(i)>0$, the effective neighborhood size is

$$
N_{\mathrm{eff}}(i)=
\exp\left(-\sum_{j\in\mathcal N_{100}(i)}p_{ij}\log p_{ij}\right).
$$

Set $N_{\mathrm{eff}}(i)=0$ when $W(i)=0$. $W(i)$ measures total connection
strength, while $N_{\mathrm{eff}}(i)$ is the entropy-equivalent number of
neighbors sharing that strength. The calculation yields one pair
$(W,N_{\mathrm{eff}})$ in statement space and another in proof space, with
$N_{\mathrm{eff}}\leq100$. Threshold quantiles from 0.975 to 0.995 are included
as a rank-stability check.

**Interpretation:** high strength and effective size indicate membership in a
broad, dense semantic community; low strength indicates isolation.

## 3. Structural proof complexity

**Intuition:** How involved is its recorded proof?

For target $i$, derive the following directly from the traced proof and the
dependency graph:

- $s_i$: number of entries in `traced_tactics` (also stored as `n_tactics` in
  the embedding manifest);
- $r_i$: number of distinct direct kernel constants in the target's type and
  proof term;
- $h_i$: maximum upstream dependency depth;
- $H_i=-\sum_t q_{it}\log q_{it}$: tactic-head entropy, where $q_{it}$ is the
  fraction of the proof's steps whose tactic text begins with tactic head $t$.

The tactic head is extracted with the repository's existing rule: the first
Lean identifier in each step's `tactic` string, or `<anon>` if none is present.
After transforming skewed counts and standardizing each component across the
10,000 targets, define

$$
C(i)=a\,z\!\left(\log(1+s_i)\right)
    +b\,z\!\left(\log(1+r_i)\right)
    +c\,z(h_i)+d\,z(H_i),
$$

where $z(y)=(y-\bar y)/\operatorname{sd}(y)$. Initial weights may be equal, but
they must be frozen in advance; learned weights would require an explicit
external target such as proof-generation success and separate validation.

**Interpretation:** larger values mean greater observed structural involvement,
not necessarily greater mathematical difficulty.

## Reproduce

The dependency exporter uses the pinned Lean runtime prepared by the earlier
generation experiment. From the repository root:

```powershell
python experiments/theorem-network-measures/scripts/analyze.py
python experiments/theorem-network-measures/scripts/build_report.py
```

The first command caches a large, gitignored kernel-dependency export, computes
all scores, and writes `artifacts/` and `figures/`. The second renders the compact
MathJax report at [`index.html`](index.html). See [`RESULTS.md`](RESULTS.md) for
the findings and `config.json` for frozen settings.
