# Archived neighborhood-thinning cosmology synthesis

This directory preserves the completed five-page paper and the scientific
checkpoint from 2026-08-21. Together they synthesize the first DESI,
causal-volume, and CAMELS tests. The archive remains reproducible, but its
local-neighborhood interpretation is no longer the active accumulated-history
program.

- `neighborhood_thinning_cosmology.tex` and `.pdf` — paper source and compiled
  paper.
- `SCIENTIFIC_CHECKPOINT_2026-08-21.md` — full interpretation and recorded
  results at the time of completion.
- `make_paper_figures.py` and `figures/` — frozen figure generation and outputs.

The underlying live experiments now reside in:

- `../../accumulated_history/experiments/expansion_history/`
- `../../accumulated_history/experiments/causal_volume/`
- `../../spatial_neighborhoods/binding/`

Build the archived paper from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error neighborhood_thinning_cosmology.tex
```
