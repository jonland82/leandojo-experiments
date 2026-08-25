# Noether-compatible histories note

This folder contains the single-column sequel to `../note/`. It develops the
accumulated-history program from geometric neighborhood survival to
Noether-defined admissibility and invariant Liouville measure.

- `noether_compatible_histories.tex` — LaTeX source
- `noether_compatible_histories.pdf` — compiled note
- `radius_identifiability.pdf` and `measure_invariance.pdf` — vector figures
- `make_figures.py` — regenerates both figures from the experiment code

The shared bibliography is at `../../references/complexity_cosmology.bib`.

Build from this directory with:

```powershell
python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error noether_compatible_histories.tex
```
