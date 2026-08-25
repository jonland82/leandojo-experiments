# Controlled estimator-validation results

Run on 2026-08-24 with NumPy on the local laptop. The final run used seed
`20260824`, 12 primitive binary updates, and ensembles from 100,000 through
2,000,000 paired histories.

## Results

- **Predictable Hubble flow:** raw fixed-radius complexity increased from
  1.0000 bits in the static null to 1.9546 bits under predictable stretching.
  Exact dynamical conditioning reduced it to 0 bits.
- **Deterministic chaos:** raw trajectory divergence produced 0.5581 bits at
  the fixed test radius. Exact conditioning again reduced it to 0 bits.
- **Injected random distinctions:** the largest ensemble recovered
  1.0022 bits per primitive update. Regrouping two updates per reporting
  interval recovered 1.0010 bits per primitive update.
- **Fixed-radius tolerance:** at update 12, empirical complexities for mismatch
  budgets `r=0,1,2` were 11.9687, 8.3247, and 5.6973 bits, versus exact values
  12.0000, 8.2996, and 5.6962 bits.
- **Deterministic hidden state:** conditioning only on the revealed macro-past
  recovered 0.9971 bits per update; conditioning on the complete initial
  microstate reduced the novelty to 0 bits.

All eight predeclared validation checks passed in the final run.

## Sampling note

The first run stopped at 1,000,000 pairs. Its rare `r=0` bin contained 212
survivors against 244 expected and missed the fixed 0.20-bit tolerance by
0.0036 bits. The ensemble ceiling was doubled without changing the estimator
or acceptance threshold. The 2,000,000-pair run passed, consistent with the
initial miss being rare-event sampling noise.

## Interpretation

The estimator now clears its synthetic gate: it does not count exactly
predictable expansion or deterministic chaos as conditional new information,
and it recovers known distinction rates across ensemble size, tolerance, and
time grouping. This validates the measurement procedure only. It does not yet
show that an independently measured history-survival rate predicts cosmological
expansion.
