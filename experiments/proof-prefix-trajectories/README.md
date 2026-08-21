# Proof-prefix trajectories

This experiment tests a geometric mechanism for proof-space isolation. It holds
the 10,000-proof reference corpus and its cosine threshold fixed, then follows
the embedding of 300 proofs with at least 16 traced tactics through prefixes of
1, 2, 4, 8, and 16 steps and through the full proof. A matched control repeats
the first tactic to each length, increasing text length without adding tactic
diversity.

The primary comparison is the within-proof change from 1 to 16 tactics. The
analysis measures weighted degree, effective neighborhood size, neighbor count,
retention of the initial top-100 neighbors, and residual energy outside their
linear span. Exact duplicate inputs at checkpoint 1 and re-embedded full proofs
provide embedding reliability checks.

Run from the repository root:

```powershell
python experiments/proof-prefix-trajectories/scripts/prepare.py
python experiments/proof-prefix-trajectories/scripts/embed_aws_cli.py
python experiments/proof-prefix-trajectories/scripts/analyze.py
```

See `RESULTS.md` for the findings and `config.json` for the frozen design.
