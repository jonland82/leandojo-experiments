# Project-vocabulary ablation

This experiment tests whether the residual separation of frontier statements is
a repository-vocabulary fingerprint. Starting from the alpha-normalized view,
it replaces lexically identifiable project-defined global constants with
statement-local placeholders `g0`, `g1`, ... while preserving repeated-symbol
identity, Mathlib vocabulary, syntax, and binders.

Resolution is conservative: exact qualified/raw project declaration names are
replaced, as are unqualified short names only when unique within that project.
Ambiguous short names are retained to avoid accidentally deleting Mathlib
vocabulary. Reference and exact-snapshot control texts are unchanged, so their
existing embeddings are reused; only 18,852 frontier statements are re-embedded.

This is a lexical sensitivity analysis, not elaborator-certified symbol
resolution. The AWS cost guard inherits the existing $15 cap.

