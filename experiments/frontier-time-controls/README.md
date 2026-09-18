# Time-matched controls for frontier formalizations

This experiment asks whether the statement-space isolation of major Lean
formalizations survives comparison with ordinary Mathlib declarations from the
same period and toolchain. For each of eight modern projects, it samples
theorem and lemma statements from the exact Mathlib revision pinned by that
project. Statements already present verbatim in the February 2024 LeanDojo
corpus are excluded, so the controls represent changed or newly introduced
Mathlib material rather than exact historical carryovers.

Every same-era control and frontier declaration is mapped into the original
frozen 10,000-item LeanDojo statement space. Within each project, frontier
declarations are matched to controls from its pinned snapshot, exactly on the
frozen semantic cluster and nearest in log statement length. The primary
summary gives equal weight to projects.

`lean-liquid` is excluded from the primary analysis: it uses Lean 3.48 and a
January 2024 Mathlib 3 revision, while the frozen reference is Mathlib 4 from
February 2024. Its earlier comparison is already approximately time-matched,
but not toolchain-matched.

Run from the repository root:

```powershell
python experiments/frontier-time-controls/scripts/prepare.py
python experiments/frontier-time-controls/scripts/embed_aws_cli.py
python experiments/frontier-time-controls/scripts/analyze.py
```

The Mathlib history clone lives under ignored `data/`. Embedding is statement
only and uses the global Cohere Embed v4 inference profile; the preflight cost
guard refuses calls above $15.
