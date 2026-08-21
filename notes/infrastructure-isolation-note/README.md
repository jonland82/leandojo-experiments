# Infrastructure–isolation companion note

Build from the repository root after completing the `theorem-network-measures`
and `proof-prefix-trajectories` experiments:

```powershell
python notes/infrastructure-isolation-note/make_figure.py
python notes/infrastructure-isolation-note/make_trajectory_figure.py
cd notes/infrastructure-isolation-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The generated `note.pdf` is a standalone companion to the earlier notes, using
the same US Letter layout, typography, narrative style, and level of
mathematical detail. It preserves the rank-sign proposition and tetrad test,
and adds a conditional orthogonal-extension proposition tested with proof-prefix
trajectories.
