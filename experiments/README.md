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
| [`theorem-network-measures/`](theorem-network-measures/) | 10,000 theorem–proof pairs | Per-theorem dependency, connectedness, and complexity measures | Complete |
| [`proof-prefix-trajectories/`](proof-prefix-trajectories/) | 300 long proofs, 3,506 trajectories | Within-proof semantic-diversification mechanism test | Complete |
| [`frontier-formalizations/`](frontier-formalizations/) | 18,852 declarations from 9 projects | Locate major recent formalizations in the frozen LeanDojo space | Complete |
| [`frontier-rarity-controls/`](frontier-rarity-controls/) | 18,852 frontier + 10,000 reference declarations | Test whether length and coarse domain explain frontier isolation | Complete |
| [`frontier-time-controls/`](frontier-time-controls/) | 18,852 frontier + 21,000 exact-snapshot controls | Test whether recency and toolchain drift explain frontier isolation | Complete |
| [`frontier-style-ablation/`](frontier-style-ablation/) | 49,852 statements across 3 cohorts | Test whether headers and bound-variable names explain residual isolation | Complete |

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
