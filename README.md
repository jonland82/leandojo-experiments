# LeanDojo Proof Geometry Experiments

An empirical study of how theorem statements, human-written Lean proofs, and
proof-generation behavior relate across LeanDojo Benchmark 4.

The repository follows a single research arc:

1. separate **proof procedure** (tactic style) from **mathematical subject**
   (explicit premise vocabulary);
2. compare statement and proof representations in learned semantic spaces;
3. test whether nearby statements have nearby recorded proofs; and
4. intervene by retrieving examples before generation and checking every
   candidate with Lean.

[Explore the project site](https://jonland82.github.io/leandojo-experiments/) ·
[Open the 3-D proof-space viewer](https://jonland82.github.io/leandojo-experiments/app/) ·
[Read the experiment index](experiments/)

## Results at a glance

- At 10,000 proofs, tactic-style and premise-domain topics have only weak
  alignment: adjusted mutual information (AMI) is **0.0153** and Cramér's V is
  **0.102**.
- Statement-only and proof-only semantic clusters also differ globally
  (**AMI 0.0804**), but local geometry is informative: at `k = 10`, statement
  neighbors' proofs are **0.0696 cosine points** closer than a context-matched
  random baseline.
- In the paired 100-target generation pilot, semantic retrieval reached
  **24% pass@3**, versus 21% for BM25, 16% with no retrieval, and 14% with
  random examples. The pilot supports useful retrieval, but does not establish
  that semantic retrieval is better than BM25.

These are descriptive and pilot-scale findings, not claims that proof spaces
have a canonical number of clusters or that theorem statements determine
proofs. The full reports retain uncertainty estimates, controls, and caveats.

## Repository guide

| Path | Contents |
|---|---|
| [`experiments/`](experiments/) | Frozen designs, scripts, inputs, outputs, and results for five completed experiments |
| [`notes/`](notes/) | Four compact LaTeX/PDF research notes and their figure-generation code |
| [`app/`](app/) | Dependency-free interactive 3-D viewer for the original 1,940-proof analysis |
| [`pipeline.py`](pipeline.py) | Shared style/domain topic pipeline for the 1,940- and 10,000-proof profiles |
| [`scripts/`](scripts/) | Cross-view analysis and small artifact/reporting utilities |
| [`FINDINGS.md`](FINDINGS.md) | Full write-up of the original style/domain experiment |
| `out/` | Historical artifacts consumed by the original viewer and reporting utilities |

Large arrays and raw model/verifier outputs are checked in intentionally so the
published analyses can be inspected without rerunning paid services. The
LeanDojo benchmark itself is not redistributed; `data/` is local and ignored.

## Quick start

Python 3.13 was used for the recorded local runs. Create an environment and
install the pinned analysis dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place LeanDojo Benchmark 4 at:

```text
data/leandojo_benchmark_4/leandojo_benchmark_4/
```

Then reproduce either tactic-style/domain profile from the repository root:

```powershell
python pipeline.py --profile small-1940
python pipeline.py --profile aws-10000
```

The larger semantic and generation studies have additional AWS, model-access,
or Lean/Mathlib requirements. Their local READMEs contain exact commands and
frozen configuration:

- [`semantic-embeddings-10000`](experiments/semantic-embeddings-10000/)
- [`semantic-neighborhood-transfer-10000`](experiments/semantic-neighborhood-transfer-10000/)
- [`retrieval-guided-proof-generation-100`](experiments/retrieval-guided-proof-generation-100/)

## Explore without installing anything

Open [`app/index.html`](app/index.html) directly in a browser. Its data is
embedded through `app/data.js`, so the viewer works from `file://` and has no
CDN or server dependency. It supports theorem/file search, PCA and t-SNE
layouts, proof-length sizing, and style/domain coloring.

The repository root [`index.html`](index.html) is the GitHub Pages landing page.
It can be previewed locally by opening the file; MathJax is loaded from jsDelivr
for the small amount of typeset mathematics.

## Reproducibility boundaries

- Randomized local analyses use recorded seed 0 unless an experiment says
  otherwise.
- Semantic embeddings and raw generation responses are retained with checksums
  and provenance metadata.
- Bedrock generation at temperature `0.4` is not bit-for-bit reproducible
  because the API did not expose a sampling seed.
- Lean acceptance was checked against the pinned historical Mathlib context;
  setup instructions live with the generation experiment.
- Generated PDFs are checked in alongside their sources for convenient review.

## Project status

All five documented experiments and four notes are complete. The natural next
step is a larger paired comparison of semantic retrieval, BM25, and no
retrieval; that run has not been performed in this repository.

No open-source license has been selected yet. Add one before inviting external
reuse or contributions.
