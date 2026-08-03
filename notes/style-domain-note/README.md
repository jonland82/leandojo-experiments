# Style/domain note

Build from the repository root:

```powershell
python notes/style-domain-note/make_figure.py
cd notes/style-domain-note
pdflatex -interaction=nonstopmode -halt-on-error note.tex
pdflatex -interaction=nonstopmode -halt-on-error note.tex
```

The generated `note.pdf` is designed for US Letter paper at 11 pt with one-inch margins.
