# Paid follow-up: retrieval-guided Lean proof generation on AWS

> **Execution status (2026-08-04): complete.** The capped 100-target pilot was
> run in the separate sibling directory
> `experiments/retrieval-guided-proof-generation-100/`. See its `RESULTS.md` for
> kernel-verified outcomes, cost, deviations from this prospective plan, and
> reproducibility artifacts. This document is retained as the preregistered
> planning context.

## Why this follows the geometry test

The local geometry experiment asks whether statement similarity predicts proof
similarity among the single recorded proofs in LeanDojo Benchmark 4. A positive
local association would motivate a practical intervention: retrieve proofs of
nearby statements and test whether they help a generative model prove a new
theorem. The generation experiment remains worthwhile as a small pilot even if
the association is modest, because retrieved examples may contribute useful
premises or proof fragments without making complete proofs geometrically close.

The text below is the original prospective plan and is retained to make the
decision trail auditable. The later pilot did run paid Bedrock inference, with a
smaller frozen budget and local Lean verification; its `RESULTS.md` records the
actual model, candidate count, cost, runtime, and deviations.

## Geometry gate: passed locally

The completed local test supplies a positive premise for this pilot. Among the
ten nearest statement neighbors, mean proof cosine is 0.6727 versus 0.6030 for
random pairs matched on exact-file and coarse-module context, a difference of
0.0696. Statement and proof cosine have rank correlation 0.3415 across 500,000
sampled theorem pairs, and the mean top-10 neighborhood overlap is 0.1343 versus
random expectation 0.0010. See `RESULTS.md` for controls and limitations.

This clears the scientific gate for a paid pilot, but does not guarantee that
retrieval improves kernel-verified proof generation. That remains the causal,
operational question below.

## Scientific question

For a held-out statement, does semantic retrieval improve the probability that
Lean accepts a generated tactic proof relative to no retrieval, random examples,
and lexical retrieval?

The primary outcome is kernel-verified success, not similarity to the recorded
proof. For target theorem `i` and retrieval condition `c`, define

```text
success(i, c) = 1 if at least one generated proof is accepted by Lean, else 0.
```

The paired design compares conditions on exactly the same target theorems.

## Pilot design

- Freeze 100 targets from the 178 test-split records already present in the
  10,000-theorem sample.
- Restrict every retrieval bank to the 9,637 training-split records.
- Exclude the target, exact statement duplicates, and near-duplicates above a
  similarity threshold frozen before any generation.
- Evaluate four conditions:
  1. theorem statement with no examples;
  2. four randomly selected training theorem--proof examples;
  3. four lexical/BM25 statement neighbors and their proofs;
  4. four semantic statement neighbors and their proofs.
- Request four independent candidate tactic scripts per target and condition,
  giving `100 x 4 x 4 = 1,600` generations.
- Use one frozen prompt template, decoding configuration, input cap, output cap,
  retry policy, and random seed schedule across conditions.
- Report pass@1, pass@4, paired percentage-point differences, paired bootstrap
  intervals, McNemar tests, generated-proof length, token use, latency, and Lean
  verification time.

The confirmatory run should use all eligible test theorems rather than silently
reclassifying training examples as test data. The complete benchmark has 961
test records with nonempty tactic traces; embedding the additional test
statements would add negligible cost relative to generation.

## AWS execution architecture

1. Create a new isolated experiment directory and immutable JSONL manifest.
2. Place prompt batches and checkpoints in an experiment-specific S3 prefix.
3. Run the orchestrator on an EC2 CPU instance in `us-east-1` with an IAM role
   limited to the required S3 prefix, Bedrock invocation, and log delivery.
4. Invoke Bedrock through the AWS CLI, matching the transport used by the
   embedding experiment. Start with eight concurrent resumable workers and
   lower concurrency automatically on throttling.
5. Use Claude Sonnet 5 through model ID `anthropic.claude-sonnet-5`, subject to a
   preflight access check. AWS describes it as an active coding model available
   in `us-east-1`:
   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html
6. On the same EC2 instance, check out Mathlib commit
   `3c307701fa7e9acbdc0680d7f3b9c9fed9081740`, matching the dataset. Install the
   corresponding Lean toolchain and cached dependencies.
7. Verify candidates in the original theorem context so the theorem being
   proved cannot be imported or cited as an existing declaration. Cache the
   environment, parallelize verification, and preserve Lean stdout/stderr for
   every attempt.
8. Save every prompt, raw response, token count, candidate proof, verifier
   result, retry, model identifier, source commit, and checksum. Stop the EC2
   instance when the run completes.

## Cost estimate checked 2026-08-03

AWS lists promotional Claude Sonnet 5 pricing of $2 per million input tokens and
$10 per million output tokens through 2026-08-31, followed by standard pricing
of $3 and $15 respectively:
https://aws.amazon.com/bedrock/pricing/

With 1,600 generations, an average 6,000 input tokens, and an average 600 output
tokens, promotional inference is estimated as

```text
1,600 x (6,000 x $2 / 1,000,000 + 600 x $10 / 1,000,000) = $28.80.
```

Allowing for retries, longer cases, EC2 verification, storage, and logs gives a
pilot budget of **$40--$60** at promotional pricing, or **$55--$75** after the
promotion. Set an AWS Budget alert at $50 and a hard application-side token cap
before the pilot.

A confirmatory run of 500 targets under the same four-condition, pass@4 design
would require 8,000 generations. Budget approximately **$170--$220** during the
promotion or **$240--$300** afterward. Adding another retrieval condition raises
generation cost by 25 percent.

## Runtime estimate

- One-time Lean/Mathlib environment setup and cache validation: 1--3 hours.
- Pilot generation and verification after setup: 1.5--3 hours.
- Pilot from a cold EC2 instance: 3--5 hours total.
- A 500-target confirmation: approximately 8--16 hours, primarily determined
  by Bedrock account quotas and verifier concurrency.

These are planning estimates. The runner must calculate exact cost from usage
fields returned by Bedrock and update a live checkpoint after every response.

## Interpretation and stopping rule

Proceed from the pilot only if semantic retrieval improves kernel-verified
pass@4 over both no retrieval and random retrieval, and is competitive with or
better than BM25. A failure against BM25 would still be informative: it would
show that local embedding geometry does not add operational value beyond
surface lexical similarity. Analyze failures by missing premises, malformed
Lean, elaboration errors, tactic errors, and timeouts rather than collapsing all
failures into one category.
