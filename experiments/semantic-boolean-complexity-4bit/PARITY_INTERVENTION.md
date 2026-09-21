# Parity intervention analysis

For every four-input function, this analysis XORs each of the 15 nonzero
linear parities into its truth table. The decisive comparison holds the
paper's original descriptor exactly fixed and excludes input-renaming orbits.

## Matched control

| category              | language   |   pairs |   mean_absolute_gate_difference |   median_absolute_gate_difference |   q90_absolute_gate_difference |   maximum_absolute_gate_difference |   fraction_nonzero |   fraction_at_least_3 |   fraction_at_least_5 |
|:----------------------|:-----------|--------:|--------------------------------:|----------------------------------:|-------------------------------:|-----------------------------------:|-------------------:|----------------------:|----------------------:|
| parity_related        | NAND       |   17024 |                          1.2710 |                                 1 |                              3 |                                  8 |             0.7530 |                0.1246 |                0.0100 |
| parity_related        | NOR        |   17024 |                          1.2684 |                                 1 |                              3 |                                  8 |             0.7361 |                0.1169 |                0.0093 |
| parity_related        | AND_OR_NOT |   17024 |                          0.6635 |                                 1 |                              2 |                                  2 |             0.5609 |                0.0000 |                0.0000 |
| other_same_descriptor | NAND       | 8040999 |                          1.1672 |                                 1 |                              2 |                                  8 |             0.7385 |                0.0936 |                0.0036 |
| other_same_descriptor | NOR        | 8040999 |                          1.1672 |                                 1 |                              2 |                                  8 |             0.7386 |                0.0936 |                0.0036 |
| other_same_descriptor | AND_OR_NOT | 8040999 |                          0.8132 |                                 1 |                              2 |                                  5 |             0.6199 |                0.0280 |                0.0004 |

## Descriptor-preserving interventions by parity order

|   parity_order | language   |   pairs |   mean_absolute_gate_difference |   fraction_changed |   maximum_absolute_gate_difference |
|---------------:|:-----------|--------:|--------------------------------:|-------------------:|-----------------------------------:|
|              1 | AND_OR_NOT |    4464 |                          0.7222 |             0.6111 |                                  2 |
|              1 | NAND       |    4464 |                          0.9919 |             0.6828 |                                  4 |
|              1 | NOR        |    4464 |                          1.0000 |             0.6532 |                                  4 |
|              2 | AND_OR_NOT |    5640 |                          0.7021 |             0.5915 |                                  2 |
|              2 | NAND       |    5640 |                          1.4117 |             0.7989 |                                  6 |
|              2 | NOR        |    5640 |                          1.4117 |             0.7989 |                                  6 |
|              3 | AND_OR_NOT |    4464 |                          0.7168 |             0.6084 |                                  2 |
|              3 | NAND       |    4464 |                          1.2482 |             0.7634 |                                  6 |
|              3 | NOR        |    4464 |                          1.2303 |             0.7285 |                                  5 |
|              4 | AND_OR_NOT |    2456 |                          0.3713 |             0.3127 |                                  2 |
|              4 | NAND       |    2456 |                          1.4963 |             0.7561 |                                  8 |
|              4 | NOR        |    2456 |                          1.4963 |             0.7561 |                                  8 |

## Largest controlled examples

| left_truth_table_hex   | right_truth_table_hex   |   parity_order |   left_anf_linear_count |   right_anf_linear_count |   NAND_absolute_difference |   NOR_absolute_difference |   AND_OR_NOT_absolute_difference |
|:-----------------------|:------------------------|---------------:|------------------------:|-------------------------:|---------------------------:|--------------------------:|---------------------------------:|
| 0x011e                 | 0x6888                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x0136                 | 0x68a0                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x0156                 | 0x68c0                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x0316                 | 0x6a80                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x0516                 | 0x6c80                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x1116                 | 0x7880                  |              4 |                       4 |                        0 |                          7 |                         8 |                                0 |
| 0x877f                 | 0xeee9                  |              4 |                       0 |                        4 |                          8 |                         7 |                                0 |
| 0x937f                 | 0xfae9                  |              4 |                       0 |                        4 |                          8 |                         7 |                                0 |

## Interpretation

Parity-related functions are not merely nearby in the original feature
space: the matched analysis compares them inside the exact same descriptor
class. Any systematic excess gate difference therefore isolates information
discarded by bias, sorted influences, Fourier energy by degree, algebraic
degree, and stabilizer size. The ANF intervention changes only constant/linear
support, leaving every higher-degree ANF coefficient fixed.

The result is basis-relative. For NAND, parity-related pairs have mean gap
1.271 versus 1.167 in controls, and gaps of at least five occur 1.00% versus
0.36%; NOR is similar. Full four-variable parity produces the largest gaps.
For AND/OR/NOT, however, parity pairs are closer than controls (mean .664
versus .813) and never differ by more than two gates. Thus the experiment
supports phase sensitivity for NAND/NOR but falsifies parity burden as a
universal explanation. The ANF-linear count's predictive gain for AND/OR/NOT
must instead be a proxy for another property, plausibly factorability.
