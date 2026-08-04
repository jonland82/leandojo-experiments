# Semantic neighborhood transfer at 10,000 proofs

This follow-up asks a local question that the earlier clustering experiment did
not answer: when two theorem statements are close in statement-embedding space,
are their recorded proofs also close in proof-embedding space?

The experiment is isolated from `semantic-embeddings-10000`. It reads the
retained statement and proof matrices and their manifest but does not modify
them or call AWS. Outputs are written only under this directory.

## Reproduce

From the repository root:

```powershell
python experiments/semantic-neighborhood-transfer-10000/scripts/analyze_geometry.py
```

The script computes exact top-100 neighborhoods in both views, compares proof
similarity along statement neighborhoods against a coarse-module-matched random
baseline that also preserves exact-source-file status, measures cross-view
neighborhood overlap, samples global theorem pairs, and fits a controlled
descriptive regression. See `RESULTS.md` for the findings. The paid follow-up
designed in `AWS_PROOF_GENERATION_PLAN.md` is now complete under
`../retrieval-guided-proof-generation-100/`.
