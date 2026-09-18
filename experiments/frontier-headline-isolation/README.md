# Within-repository headline isolation

This local experiment asks whether the 34 modern headline theorems themselves
are more isolated than ordinary declarations from the same repositories. It
uses the project-vocabulary-anonymized embeddings and matches up to 20 controls per
headline within project and frozen raw semantic cluster on statement/proof
length, file position and size, scope, binders, imports, and local references.

The primary effect is

\[
I_H^*(t)=\overline{\rho_k(c):c\in M(t)}-\rho_k(t),
\]

so positive values mean that the headline theorem is more isolated. A strict
sensitivity analysis excludes sampled declarations lying in any headline's
lexical prerequisite cone. No AWS calls are made.
