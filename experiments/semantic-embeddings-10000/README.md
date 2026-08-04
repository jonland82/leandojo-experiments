# Semantic embeddings at 10,000 proofs

This experiment is an independent semantic-view extension of the fixed-seed
`aws-10000` experiment. It embeds three representations of exactly the same
10,000 theorem records:

1. theorem statement only;
2. complete traced tactic proof only;
3. theorem statement and proof together.

The experiment intentionally lives outside `experiments/aws-10000/`. It reads
that experiment's `proofs.json` only to verify sample alignment and to compare
the new clusters with the earlier style/domain assignments. It does not update
the earlier artifacts, `FINDINGS.md`, or the paper-facing application data.

## Reproduce

Run from the repository root with AWS CLI credentials that can invoke Cohere
Embed v4 in Amazon Bedrock:

```powershell
python experiments/semantic-embeddings-10000/scripts/prepare_inputs.py
python experiments/semantic-embeddings-10000/scripts/embed_aws_cli.py
python experiments/semantic-embeddings-10000/scripts/analyze.py
```

The embedding runner shells out to `aws bedrock-runtime invoke-model`; it does
not call Bedrock through an SDK. Completed batches are saved independently and
are reused on restart.

See `RESULTS.md` for the completed findings and `config.json` for the frozen
experimental settings.
