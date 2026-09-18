# Frontier dependency shortcuts

This local experiment tests whether named frontier results occupy unusually
compressive positions in their own source dependency graphs. For every headline
theorem it compares the theorem and its prerequisite cone with 20 ordinary
declarations from the same project, matched on proof length, source position,
file size, and declaration kind. Declarations in any headline prerequisite cone
are excluded from the control pool, preventing the final theorem from being
compared against its own mathematical machinery.

The graph is deliberately source-static: an edge is inferred when a declaration
body lexically names another project declaration. Consequently the graph is a
conservative proxy, not Lean's elaborated environment. Exact qualified names,
globally unique short names, and earlier same-file names are resolved; generated
and notation-hidden dependencies can be missed.

The main compression proxy for a prerequisite lemma \(v\) is

\[
  S(v)=\max\{0,(d^+(v)-1)|p_v|-d^+(v)|n_v|\},
\]

where \(d^+(v)\) is direct downstream reuse, \(|p_v|\) is body length, and
\(|n_v|\) is reference-name length. Cone compression is
\(\sum S(v)/(\sum S(v)+\sum |p_v|)\). No AWS calls are made.

Run from the repository root:

```powershell
python experiments/frontier-dependency-shortcuts/scripts/prepare.py
python experiments/frontier-dependency-shortcuts/scripts/analyze.py
```
