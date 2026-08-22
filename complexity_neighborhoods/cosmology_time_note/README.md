# Accumulated-time cosmology note

This folder contains the focused single-column note connecting the
relational-shape theorem to an accumulated-history cosmology, the successful
present-day Hubble-scale estimate, its DESI tests, and the resulting dynamical
conditional-information program.

- `accumulated_time_cosmology.tex` — LaTeX source
- `accumulated_time_cosmology.pdf` — compiled six-page note

The bibliography is shared with the larger study at
`../neighborhood_thinning_cosmology/complexity_cosmology.bib`.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error accumulated_time_cosmology.tex
```
