# AWS 10,000-proof experiment

This profile takes a reproducible uniform sample, without replacement, of 10,000 proofs from all
52,187 theorem records with nonempty tactic traces in the benchmark's random train, validation,
and test splits. Sampling and model initialization both use seed 0.

Run it from the repository root:

```powershell
python pipeline.py --profile aws-10000
```

The AWS-produced artifacts live in `artifacts/`. `RESULTS.md` records the findings and compares
them with the original 1,940-proof experiment.
