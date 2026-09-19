# Results: four-input semantic prediction of exact Boolean formula complexity

## Design

All 65,536 four-input Boolean functions were synthesized exactly in NAND, NOR, and AND/OR/NOT. The predictor uses only signed bias, sorted influences, Walsh--Fourier energy by degree, algebraic degree, and input-permutation stabilizer size. It averages at least 7 nearest training functions, including the complete tied distance shell at the neighbor boundary. Complete test functions, input-permutation orbits, or exact descriptor classes are removed according to the protocol. The strict matched null permutes complexity within exact bias/degree/symmetry strata.

The universe contains 3,984 input-permutation orbits and 675 exact descriptor classes.

## Main prediction result

| Language | Function R² | Orbit R² | Descriptor R² | Descriptor ceiling R² | Strict RMSE | Null 95% interval | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| NAND | 0.890 | 0.851 | 0.777 | 0.893 | 1.505 | [0.038, 0.048] | 0.0010 |
| NOR | 0.890 | 0.851 | 0.777 | 0.893 | 1.505 | [0.038, 0.048] | 0.0010 |
| AND/OR/NOT | 0.929 | 0.906 | 0.843 | 0.931 | 1.031 | [0.047, 0.056] | 0.0010 |

## Exact synthesis scale

| Language | Maximum minimum gates | Mean minimum gates |
|---|---:|---:|
| NAND | 26 | 12.231 |
| NOR | 26 | 12.231 |
| AND/OR/NOT | 19 | 10.141 |

## Descriptor ablations under strict holdout

| Language | Features | R² | RMSE |
|---|---|---:|---:|
| NAND | full | 0.777 | 1.505 |
| NAND | bias_degree_symmetry | 0.002 | 3.188 |
| NAND | influences_only | 0.669 | 1.836 |
| NAND | fourier_only | 0.666 | 1.845 |
| NAND | without_influences | 0.637 | 1.923 |
| NAND | without_fourier | 0.791 | 1.459 |
| NOR | full | 0.777 | 1.505 |
| NOR | bias_degree_symmetry | 0.002 | 3.188 |
| NOR | influences_only | 0.669 | 1.836 |
| NOR | fourier_only | 0.666 | 1.845 |
| NOR | without_influences | 0.637 | 1.923 |
| NOR | without_fourier | 0.791 | 1.459 |
| AND/OR/NOT | full | 0.843 | 1.031 |
| AND/OR/NOT | bias_degree_symmetry | -0.059 | 2.681 |
| AND/OR/NOT | influences_only | 0.819 | 1.109 |
| AND/OR/NOT | fourier_only | 0.828 | 1.080 |
| AND/OR/NOT | without_influences | 0.636 | 1.573 |
| AND/OR/NOT | without_fourier | 0.806 | 1.149 |

## Cross-language rank stability

| Pair | Spearman rho |
|---|---:|
| NAND / NOR | 0.664 |
| NAND / AND/OR/NOT | 0.860 |
| NOR / AND/OR/NOT | 0.860 |

## Tie-aware three-bit comparison

| Language | Function R² | Orbit R² | Descriptor R² |
|---|---:|---:|---:|
| NAND | 0.803 | 0.706 | 0.538 |
| NOR | 0.803 | 0.706 | 0.538 |
| AND/OR/NOT | 0.895 | 0.822 | 0.589 |

## Robustness

### Neighbor threshold

| k | NAND R² | NOR R² | AND/OR/NOT R² |
|---:|---:|---:|---:|
| 3 | 0.778 | 0.778 | 0.845 |
| 5 | 0.778 | 0.778 | 0.845 |
| 7 | 0.777 | 0.777 | 0.843 |
| 11 | 0.777 | 0.777 | 0.844 |
| 15 | 0.773 | 0.773 | 0.841 |

### Coordinate scaling

| Scaling | NAND R² | NOR R² | AND/OR/NOT R² |
|---|---:|---:|---:|
| function_weighted_zscore | 0.777 | 0.777 | 0.843 |
| descriptor_weighted_zscore | 0.815 | 0.815 | 0.836 |
| robust_median_iqr | 0.806 | 0.806 | 0.844 |

## Interpretation

The three-bit result survives decisively over the complete four-input universe. Under this order-invariant tie-aware protocol, the strict R² values are numerically higher than in the earlier exact-seven experiment. After removing every function with the test descriptor, semantic neighborhoods explain 77.7% of exact NAND/NOR variance and 84.3% of AND/OR/NOT variance. The corresponding matched-null intervals lie near 4--6%, and no permutation reaches an observed result (`p = 1/1001` in every language).

The descriptor-class mean ceilings are 89.3% for NAND/NOR and 93.1% for AND/OR/NOT. Thus the strict out-of-class predictor recovers most, but not all, of the variance that this descriptor could possibly explain. Bias, degree, and symmetry alone explain essentially nothing under strict holdout, whereas influences alone and Fourier energy alone remain strongly predictive. Removing Fourier coordinates slightly improves the NAND/NOR kNN result; this indicates distance-metric dilution, not absence of Fourier signal, because Fourier-only prediction remains strong.

Complexity is still representation-relative: NAND versus NOR rank correlation is 0.664, while each correlates 0.860 with AND/OR/NOT. This is a complete finite-universe result for four inputs, three related bases, a hand-designed descriptor, and an order-invariant tie-aware kNN rule. The tie policy is stricter and more stable than the exact-seven rule used in the current three-bit paper, so the raw R² values should not be read as a pure sample-size comparison. Exact targets pass both the three-bit regression and all-function NAND/NOR duality checks.

## Reproduce

```powershell
python experiments/semantic-boolean-complexity-4bit/scripts/run.py
```
