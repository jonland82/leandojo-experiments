# Infrastructure–isolation companion note

Build from the repository root after completing the
`theorem-network-measures` experiment:

```powershell
python notes/infrastructure-isolation-note/make_figure.py
cd notes/infrastructure-isolation-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The generated `note.pdf` is a standalone companion to the earlier notes, using
the same US Letter layout, typography, narrative style, and level of
mathematical detail. It includes a rank-sign proposition that derives the
observed correlation geometry from a latent infrastructure coordinate, plus a
tetrad test showing that one linear rank factor cannot explain the measured
magnitudes.
