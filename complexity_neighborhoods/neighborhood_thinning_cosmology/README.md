# Neighborhood-thinning cosmology study

This folder contains the five-page cosmology paper and all of its supporting
materials. It is separate from the untouched original relational-shape paper
in the parent directory and from the focused accumulated-time note in
`../cosmology_time_note/`.

Principal files:

- `neighborhood_thinning_cosmology.tex` and `.pdf` — paper source and compiled paper
- `COSMOLOGY_HYPOTHESIS.md` — complete research checkpoint
- `complexity_cosmology.bib` — bibliography shared by both cosmology notes
- `cosmology_toy_model.py` — present-scale and turnaround calculations
- `test_history_models.py` — DESI DR2 BAO shape tests
- `test_relativistic_history.py` — causal-history calculation
- `test_conditional_information.py` — CAMELS phase-space and gravity-only tests
- `make_paper_figures.py` and `figures/` — figure source and outputs
- `data/` — DESI DR2 mean vector and covariance

Build the paper from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error neighborhood_thinning_cosmology.tex
```
