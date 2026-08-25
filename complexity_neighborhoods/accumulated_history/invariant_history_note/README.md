# Invariant compatible-histories paper

This folder contains the fourth paper in the complexity-neighborhoods arc. It
constructs a measure on complete deterministic classical histories by pushing
reduced Liouville measure through the solution map, then tests canonical,
temporal, and gauge invariance.

- `invariant_classical_histories.tex` — LaTeX source
- `invariant_classical_histories.pdf` — compiled paper
- `history_measure_invariance.pdf` — vector summary figure
- `make_figures.py` — regenerates the figure from the experiment code

The underlying experiments are in `../experiments/history_survival/`:

- `run_history_pushforward.py`
- `run_history_measure_extensions.py`
- `run_constrained_history_gauge.py`

Build from this directory with:

```powershell
python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error invariant_classical_histories.tex
```
