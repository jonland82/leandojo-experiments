# Retrieval-generation companion note

Build from the repository root after completing the isolated
`retrieval-guided-proof-generation-100` experiment:

```powershell
python notes/retrieval-generation-note/make_figure.py
cd notes/retrieval-generation-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The generated `note.pdf` is a standalone three-page companion to the first two
notes, using the same US Letter layout, typography, writing style, and level of
mathematical detail.
