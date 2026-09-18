# Synthesis depth and semantic isolation

This experiment directly tests whether dependency depth explains the remaining
alpha-normalized semantic isolation of frontier declarations. Its outcome is
the declaration-level isolation residual against an exact-snapshot Mathlib
control matched on raw semantic cluster and statement length:

\[
I^*(t)=\rho_k(c(t))-\rho_k(t).
\]

Within each project, ordinary least squares estimates the standardized partial
association of \(I^*(t)\) with log prerequisite depth and bounded dependency
compression while controlling the static source-architecture features already
used in `frontier-architecture-controls`. The reported interval treats projects,
not declarations, as the replication units.

Dependency edges remain lexical rather than Lean-elaborated. No AWS calls are
made; all embeddings are reused from prior experiments.

The ignored `artifacts/declaration_metrics.jsonl` file is a local computation
cache. Delete it to rebuild density residuals and dependency cones from source.
