# Retrieval-guided Lean proof generation (100-target pilot)

This experiment tests whether proofs retrieved from semantically similar theorem
statements improve Lean proof generation. It is the paid operational follow-up
to `../semantic-neighborhood-transfer-10000/RESULTS.md`, whose local geometry
test found that nearby statements have more-similar recorded proofs than a
context-matched baseline.

The experiment is intentionally separate from the 10,000-theorem embedding and
geometry experiments. It freezes 100 held-out test theorems, retrieves only from
training records, compares no retrieval, random retrieval, BM25 retrieval, and
semantic retrieval on the same targets, and requests three candidates per
condition. Success means that a generated candidate is accepted in the original
Mathlib source context by the pinned Lean kernel.

`config.json` is the frozen design and budget policy. `inputs/` contains the
immutable target/retrieval manifest and exact prompts. `outputs/` contains raw
Bedrock responses and usage metadata. `verification/` contains per-candidate
kernel results. `artifacts/analysis.json` and `RESULTS.md` contain the final
analysis and interpretation.

The runner invokes Bedrock exclusively through the AWS CLI. A conservative
application-side reservation and the usage counts returned by Bedrock prevent
the experiment from exceeding its $23.50 inference stop or $25 absolute ceiling.

The pilot is complete. See `RESULTS.md` for the paired outcomes and
`artifacts/analysis.json` for the machine-readable result. The observed token
cost was $4.4236 and the conservative accounted total was $4.5596.

On a clean Windows machine, run `scripts/setup_lean.ps1` before verification.
It installs the pinned Lean toolchain and Mathlib checkout inside this
experiment's ignored `.runtime/` directory. It deliberately uses the committed
Lake dependency manifest rather than updating historical dependencies.
