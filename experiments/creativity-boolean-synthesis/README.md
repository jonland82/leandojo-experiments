# Exact creativity test: shortest Boolean synthesis

This experiment exactly counts ordered binary NAND formulas over three input
variables by gate count and truth-table semantics. Dynamic programming covers
all 256 Boolean functions without enumerating billions of syntax trees.

For target function \(f\) and formula \(e\),

\[
S_f(e)=\mathbf 1[\llbracket e\rrbracket=f],\qquad
G_f=L_{\mathrm{DNF}}(f)-|e|,\qquad
R_f=-\log_2 P(\llbracket e\rrbracket=f\mid |e|=L_f^*),
\]

where \(L_f^*\) is the exactly synthesized minimum and
\(L_{\mathrm{DNF}}\) is the smaller of fixed, balanced, no-sharing NAND
compilations of canonical DNF\((f)\) and NOT-DNF\((\neg f)\). The score
\(K_f=S_fG_fR_f\) verifies the three operational
components only relative to this language, baseline compiler, and prior.

Run from the repository root:

```powershell
python experiments/creativity-boolean-synthesis/scripts/analyze.py
```
