# Frontier proof-space note

This directory contains the compact paper-style report synthesizing the
frontier formalization, rarity, time, style, architecture, dependency,
depth-isolation, vocabulary-ablation, and headline-isolation experiments.

Build from this directory with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error note.tex
```

Figures are read directly from the experiments' checked-in artifacts so the
note and analyses use the same rendered objects.
