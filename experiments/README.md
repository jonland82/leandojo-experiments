# Experiment runs

Each experiment has its own directory and artifact set. The profiles are defined in
`pipeline.py` and use random seed 0 throughout.

| Directory | Selection | Purpose |
|---|---|---|
| `small-1940/` | Every tactic-traced proof in random `val` and `test` | Original exploratory experiment |
| `aws-10000/` | Fixed-seed uniform sample of 10,000 tactic-traced proofs from random `train`, `val`, and `test` | Larger-scale replication |

Run either profile from the repository root:

```powershell
python pipeline.py --profile small-1940
python pipeline.py --profile aws-10000
```

Artifacts are written to `experiments/<profile>/artifacts/`. The historical root-level `out/`
and `app/data.js` remain the original 1,940-proof artifacts used by the note and viewer.

Recompute the projection-free cross-view statistics for either artifact with:

```powershell
python analyze_cross_view.py experiments/small-1940/artifacts/proofs.json
python analyze_cross_view.py experiments/aws-10000/artifacts/proofs.json
```
