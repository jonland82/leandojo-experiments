# Original 1,940-proof experiment

This profile retains every theorem with a nonempty tactic trace from the random validation and
test splits: 979 validation proofs and 961 test proofs.

Run it from the repository root:

```powershell
python pipeline.py --profile small-1940
```

The canonical rerun artifacts are written to `artifacts/`.
[`FINDINGS.md`](../../FINDINGS.md) contains the original full write-up. The
historical root-level `out/` and `app/data.js` are retained for the published
viewer and original reporting utilities; they are intentionally not treated as
the canonical rerun output.
