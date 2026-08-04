# Semantic-embeddings companion note

Build from the repository root after completing the isolated
`semantic-embeddings-10000` experiment:

```powershell
python notes/semantic-embeddings-note/make_figure.py
cd notes/semantic-embeddings-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The generated `note.pdf` is a standalone three-page companion to
`notes/style-domain-note/note.pdf`, using the same US Letter layout, typography,
and level of mathematical detail.
