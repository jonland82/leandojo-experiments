# Experiments

The repository follows one question from descriptive proof structure to a
kernel-verified generation pilot: what information about a proof is shared with
its theorem statement, and can that shared structure improve proof generation?

## Experiment map

| Directory | Scale | Role | Status |
|---|---:|---|---|
| [`small-1940/`](small-1940/) | 1,940 proofs | Original tactic-style/domain topic analysis | Complete |
| [`aws-10000/`](aws-10000/) | 10,000 proofs | Fixed-seed large-sample replication | Complete |
| [`semantic-embeddings-10000/`](semantic-embeddings-10000/) | 10,000 theorem–proof pairs | Statement, proof, and joint embedding comparison | Complete |
| [`semantic-neighborhood-transfer-10000/`](semantic-neighborhood-transfer-10000/) | 10,000 theorem–proof pairs | Local cross-view neighborhood test | Complete |
| [`retrieval-guided-proof-generation-100/`](retrieval-guided-proof-generation-100/) | 100 held-out targets | Paired retrieval-guided generation pilot with Lean verification | Complete |

The first two profiles are defined in the root [`pipeline.py`](../pipeline.py)
and use random seed 0 throughout:

```powershell
python pipeline.py --profile small-1940
python pipeline.py --profile aws-10000
```

Artifacts are written to `experiments/<profile>/artifacts/`. The root-level
`out/` and `app/data.js` are preserved historical artifacts for the original
1,940-proof write-up and interactive viewer.

Recompute projection-free cross-view statistics with:

```powershell
python scripts/analyze_cross_view.py experiments/small-1940/artifacts/proofs.json
python scripts/analyze_cross_view.py experiments/aws-10000/artifacts/proofs.json
```

Each later experiment is self-contained and documents its own frozen settings,
commands, artifacts, and deviations. Read its `README.md` for reproduction and
its `RESULTS.md` for interpretation.
