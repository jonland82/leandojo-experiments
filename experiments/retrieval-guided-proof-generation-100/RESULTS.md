# Retrieval-guided Lean proof generation: 100-target AWS pilot

## Result in one paragraph

Semantic retrieval produced the highest kernel-verified proof rate in this
paired pilot. At least one of three generated candidates was accepted for 24 of
100 held-out theorems under semantic retrieval, compared with 21 for BM25
retrieval, 16 with no retrieved examples, and 14 with random examples. The
semantic-minus-random difference was +10 percentage points, with a 95% paired
target-bootstrap interval of +3 to +18 points and a two-sided exact McNemar
value of `p = 0.021`. Semantic retrieval's +8-point difference from no retrieval
and +3-point difference from BM25 were positive but not resolved by this pilot:
their intervals were respectively -1 to +17 points (`p = 0.134`) and -3 to +9
points (`p = 0.508`). The experiment therefore supplies operational evidence
that relevant retrieval helps, but not evidence that semantic retrieval is
better than a strong lexical baseline.

## What was tested

The local geometry experiment in
`../semantic-neighborhood-transfer-10000/RESULTS.md` found that the recorded
proof traces of nearby statements were more similar than a context-matched
random baseline. This experiment tests whether that association transfers to a
practical intervention: retrieving proof examples before asking a generative
model to prove a new theorem.

The design froze 100 theorems from the 178 test-split records in the existing
10,000-theorem sample. Every retrieval bank contained only the 9,637 sampled
training records. Exact statement duplicates and candidates with statement
cosine similarity above `0.98` were excluded. Training declarations longer than
3,000 characters were also excluded from prompt eligibility, leaving 9,592
records. Each target was evaluated under four conditions:

1. the target statement alone (`no_retrieval`);
2. four random training declarations (`random`);
3. four BM25 statement neighbors (`bm25`); and
4. four semantic statement neighbors (`semantic`).

BM25 is a lexical ranking function that rewards query terms occurring often in
a candidate statement but rarely across the retrieval bank, with a correction
for statement length. Semantic retrieval ranks the cosine similarity of the
normalized statement embeddings from the earlier AWS embedding experiment. If
`x` and `y` are nonzero embedding vectors, their cosine similarity is
`(x · y) / (||x||₂ ||y||₂)`, where `x · y` is their dot product and `||x||₂` is
the Euclidean length, the square root of the sum of squared coordinates.

The model generated three candidates for each target and condition, giving
`100 x 4 x 3 = 1,200` responses. A condition passes at `k` for a target if Lean
accepts at least one of its first `k` candidates for that target. Every candidate
was inserted after `by` in the original theorem declaration and checked in its
original source context at Mathlib commit
`3c307701fa7e9acbdc0680d7f3b9c9fed9081740` with Lean `4.6.0-rc1`. Kernel
acceptance, not similarity to the dataset proof, is the outcome.

## Main outcomes

| condition | pass@1 | pass@2 | pass@3 | accepted attempts / 300 |
|---|---:|---:|---:|---:|
| no retrieval | 11% | 14% | 16% | 38 (12.7%) |
| random examples | 13% | 13% | 14% | 33 (11.0%) |
| BM25 examples | 18% | 20% | 21% | 52 (17.3%) |
| semantic examples | **19%** | **23%** | **24%** | **61 (20.3%)** |

The 95% Wilson intervals for the individual pass@3 rates are 10.1--24.4% for no
retrieval, 8.5--22.1% for random retrieval, 14.2--30.0% for BM25, and
16.7--33.2% for semantic retrieval. These marginal intervals are included to
show sampling uncertainty in each rate; the paired differences below are the
more relevant comparisons because every condition used the same targets.

| paired pass@3 comparison | difference | paired bootstrap 95% interval | discordant targets | exact McNemar `p` |
|---|---:|---:|---:|---:|
| semantic - no retrieval | +8 points | -1 to +17 | 15 semantic only, 7 no-retrieval only | 0.134 |
| semantic - random | **+10 points** | **+3 to +18** | 13 semantic only, 3 random only | **0.021** |
| semantic - BM25 | +3 points | -3 to +9 | 6 semantic only, 3 BM25 only | 0.508 |

For the bootstrap interval, a target theorem is the resampling unit: 100 targets
are sampled with replacement 10,000 times and the paired pass-rate difference
is recomputed. The exact McNemar calculation conditions on targets where the two
conditions disagree and tests whether either direction of disagreement is
equally likely. No correction for the three pilot comparisons is applied, so
the `p` values should be read as descriptive evidence rather than confirmatory
claims.

## Reading the result

The cleanest conclusion is that retrieval quality matters. Random examples did
not help: their pass@3 rate was slightly below no retrieval, they consumed the
most output tokens, and 38 of their 300 responses reached the output cap. In
contrast, both relevance-based methods improved point estimates over no
retrieval, and semantic retrieval was best at every value of `k` reported.

The result connects the two experiments without equating them. The geometry
test established a local association between statement and proof-trace
similarity. The present intervention shows that semantic statement neighbors
can carry useful proof information into generation. It does not show that the
model copies a geometrically nearby proof, nor that the embedding space has
identified a unique proof strategy.

Semantic retrieval has not yet earned a claim of superiority over BM25. Their
point estimates differ by only three targets, and 18 targets were solved by both
conditions. The paired interval permits a small semantic disadvantage as well
as a moderate advantage. The semantic and BM25 prompts were related but not
identical: they shared 1.27 of four examples on average, and 24 targets had no
retrieved example in common. A larger paired run is therefore scientifically
reasonable, but this pilot does not justify describing the semantic method as
better than lexical retrieval.
## Model and proof representation

The planned Claude Sonnet 5 profile appeared active in the Bedrock catalog but
returned `AccessDeniedException` for this AWS account. Before the batch began,
the experiment substituted the authorized global Claude Sonnet 4.5 profile,
`global.anthropic.claude-sonnet-4-5-20250929-v1:0`. All 1,200 recorded responses
use that one model, temperature `0.4`, and a 500-token output limit.

One representation issue was also resolved before the batch. LeanDojo's
`traced_tactics` field is an execution trace; for term proofs and equation-style
proofs, concatenating those tactics is not necessarily a standalone script from
the theorem's initial goal. The earlier geometry experiment correctly studied
those traces as text, but this generation experiment uses the exact complete
source declaration for every retrieved training example. Generated outputs are
still tactic scripts intended to follow `by`, and the kernel judges them from
the target theorem's initial goal.

## Failures and sensitivity checks

Of 1,200 candidates, 184 were accepted. The remaining 1,016 rejections were
assigned one primary diagnostic category from Lean output:

| primary outcome | attempts |
|---|---:|
| unknown identifier or constant | 268 |
| unsolved goals | 252 |
| tactic failure | 144 |
| elaboration or type error | 110 |
| other Lean rejection | 91 |
| output truncated at 500 tokens | 83 |
| syntax or parse error | 56 |
| verifier timeout | 12 |

None of the 83 output-capped responses passed. Raising the cap could recover
some long proofs, but it would change both cost and decoding behavior and was
not done after seeing outcomes.

All 12 timeouts were the three candidates under all four conditions for one
theorem, `NumberField.mixedEmbedding.minkowskiBound_pos`, whose source context
was unusually expensive to compile. Excluding that theorem leaves 99 paired
targets and does not change any success count: pass@3 becomes 16.2% for no
retrieval, 14.1% for random, 21.2% for BM25, and 24.2% for semantic retrieval.
Thus the ranking and paired success differences are not artifacts of timeout
imbalance.

The diagnostic categories are coarse and assigned by the first matching error
class. They should guide future engineering, not be treated as a taxonomy of
mathematical failure.

## Cost and runtime

Bedrock returned usage for 817,170 input tokens and 131,471 output tokens,
including the successful access checks. At the frozen global Sonnet 4.5 rates
of $3 per million input tokens and $15 per million output tokens, the observed
token cost was **$4.4236**. One Windows CLI encoding failure may have occurred
after Bedrock completed six unobserved attempts. The ledger conservatively
charged every one of those attempts at its maximum possible cost, adding
$0.1360 and giving an accounted upper estimate of **$4.5596**. This is well below
the $23.50 application stop and $25 absolute ceiling.

The completed generation batch took 1,364 seconds (22.7 minutes). The 1,200
local Lean checks took 4,353 seconds of wall time (72.5 minutes) with eight
workers. The measured post-setup experiment therefore took about **95 minutes**.
Lean/Mathlib installation and cache preparation were one-time local setup costs.
No EC2 instance or paid S3 workflow was used: Bedrock was invoked through the
AWS CLI from the laptop, and kernel verification was local.

## Reproducibility record

- `config.json` freezes sampling, retrieval, model, decoding, budget, and
  verifier settings.
- `inputs/targets.jsonl` stores the selected targets, exact source declarations,
  and retrieved record identifiers.
- `inputs/prompts.jsonl` stores every exact prompt and SHA-256 digest.
- `outputs/responses.jsonl` stores every raw Bedrock response, token count,
  latency, retry, request digest, and response digest.
- `verification/attempts.jsonl` stores extracted tactics, kernel outcomes,
  stdout, stderr, elapsed time, and aligned response digests.
- `artifacts/preparation.json`, `artifacts/generation_run.json`, and
  `artifacts/analysis.json` contain checksums and machine-readable summaries.
- `scripts/prepare.py`, `scripts/generate_aws_cli.py`, `scripts/verify.py`, and
  `scripts/analyze.py` reproduce the four stages.

The target and retrieval choices are deterministic. Model sampling at
temperature `0.4` is not bit-for-bit reproducible because the Bedrock Converse
API did not expose a seed; the exact realized responses are therefore retained
as first-class artifacts.

## Decision

This pilot cleared the practical gate for a larger paired study: semantic
retrieval beat random retrieval and produced the best observed pass@3 rate.
The next study should focus on semantic versus BM25 and no retrieval, increase
the held-out target count, preclassify slow source contexts, and decide in
advance whether to raise the output limit. Until then, the defensible claim is
that semantic retrieval is useful and competitive with BM25—not that it is
superior to BM25.
