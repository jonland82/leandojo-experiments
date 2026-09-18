# Conditional rarity of frontier formalizations

This local experiment asks whether the density deficit of the nine frontier
formalization projects survives two observable controls: text length and coarse
semantic domain. It reuses the frozen embeddings, clusters, and exact
neighborhoods already produced by the 10,000-item LeanDojo experiments and the
frontier mapping. It makes no AWS calls.

Two estimators are reported for statement and proof density:

1. HC3-robust linear models, adding length covariates and statement-cluster
   fixed effects in stages.
2. Nearest-neighbor matching with replacement, exact on the frozen statement
   cluster and nearest in standardized log statement/proof character counts.

The statement analysis is primary. Frontier proof bodies were extracted from
source, whereas the LeanDojo proof view contains traced tactics, so proof-space
effects are descriptive. Project age/toolchain and intrinsic mathematical
complexity are not identified by this design. The declaration-level HC3
intervals condition on the sampled projects; the project-balanced matching
interval is the preferred summary of variation across projects.

This experiment depends on the prepared records from
`experiments/frontier-formalizations/`. If those ignored embedding caches are
not present in a fresh clone, run that experiment's preparation and embedding
steps first.

Run from the repository root:

```powershell
python experiments/frontier-rarity-controls/scripts/analyze.py
```
