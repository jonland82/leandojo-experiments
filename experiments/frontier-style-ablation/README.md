# Frontier formalization style ablation

This experiment tests whether surface conventions account for the residual
statement-space isolation that remains after exact-snapshot time matching. It
re-embeds the same 10,000 LeanDojo references, 21,000 changed/new Mathlib
controls, and 18,852 frontier declarations under two deterministic transforms:

1. `header_normalized` removes declaration attributes, modifiers, the
   `theorem`/`lemma` keyword, and declaration name, then canonicalizes spacing.
2. `alpha_normalized` additionally replaces locally bound identifiers with
   placeholders in order of appearance while retaining constants and type
   expressions.

For each representation, density is recomputed against the correspondingly
transformed reference corpus. Frontier declarations are then matched to
exact-snapshot controls using the same frozen raw semantic cluster and nearest
normalized statement length. This deliberately holds the domain strata fixed.

The alpha transform is token-based rather than Lean-elaborated so it works
across the heterogeneous project toolchains. Its output is auditable, but it
should be interpreted as a style ablation rather than semantic equivalence.

Run from the repository root after `frontier-time-controls`:

```powershell
python experiments/frontier-style-ablation/scripts/prepare.py
python experiments/frontier-style-ablation/scripts/embed_aws_cli.py
python experiments/frontier-style-ablation/scripts/analyze.py
```

The run uses the global Cohere Embed v4 inference profile so both transformed
views share one serving path. The preflight guard refuses AWS calls above the
existing $15 cap.
