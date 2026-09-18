# Abstraction-architecture controls

This local experiment asks how much of the alpha-normalized frontier-density
deficit survives observable source architecture. It extracts the same static
features from the eight modern frontier projects and their exact pinned
Mathlib controls: statement/proof size, file/module size, declaration position,
direct imports, lexical scope depth, binder structure, delimiter depth, the
prior definition/theorem mix, and approximate reuse of earlier local lemmas.

Within each project, features are standardized against its exact-snapshot
controls and compressed to at most six principal components explaining at
least 80% of control variance. Matching is exact on the frozen raw semantic
cluster and nearest in this architecture representation. A fixed-effect
regression provides a separate adjustment check.

This is source-static and cross-toolchain; it does not claim elaborated Lean
dependency graphs. Local reference reuse is lexical and may miss qualified or
generated declarations. The experiment makes no AWS calls.

Run from the repository root after the time-control and style-ablation stages:

```powershell
python experiments/frontier-architecture-controls/scripts/prepare.py
python experiments/frontier-architecture-controls/scripts/analyze.py
```
