# Proof-geometry literature note

This standalone fourth companion note relates the preceding three experimental
notes to proof theory, formal-library networks, representation learning, and
retrieval-guided theorem proving.

Build from the repository root:

```powershell
python notes/proof-geometry-literature-note/make_figure.py
cd notes/proof-geometry-literature-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The figure script reads the archived results of all three experiments. The
generated `note.pdf` uses the same US Letter layout and typography as the first
three notes and is limited to five pages.
