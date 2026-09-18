# Mapping frontier formalizations into LeanDojo proof space

This experiment places declarations from major standalone Lean formalizations
into the frozen Cohere Embed v4 spaces constructed by
`../semantic-embeddings-10000/`. It keeps the 10,000 LeanDojo reference sample
fixed and measures each frontier declaration's nearest reference neighbors,
local similarity, cluster assignment, and statement/proof neighborhood overlap.
It also renders standalone statement and proof maps plus a vertically stacked
2D comparison for the paper; every PCA coordinate system is fit only on the
frozen reference corpus.

The source repositories are shallow-cloned under the ignored
`data/frontier_sources/` directory. Every theorem and lemma is retained for the
tractable projects. The much larger FLT repository is represented by a
deterministic sample plus its named headline declarations.

The extractor is deliberately source-based: it supports heterogeneous Lean
versions without requiring all projects to share a toolchain. Consequently,
the statement view is the primary cross-project comparison; proof bodies are a
secondary textual view rather than LeanDojo-equivalent tactic traces.

Run from the repository root:

```powershell
python experiments/frontier-formalizations/scripts/prepare.py
python experiments/frontier-formalizations/scripts/embed_aws_cli.py --views statement proof
python experiments/frontier-formalizations/scripts/analyze.py
```

`embed_aws_cli.py` refuses to call AWS when the conservative projected charge
exceeds the hard $15 cap in `config.json`.
