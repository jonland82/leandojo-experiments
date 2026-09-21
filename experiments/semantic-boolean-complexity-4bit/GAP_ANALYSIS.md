# Descriptor-gap analysis

This analysis asks which invariant semantic summaries split functions that
the current descriptor aliases despite different exact formula costs. Ceiling
gains are diagnostic; they are not held-out prediction scores.

## Scale

- Current descriptor classes: 675
- Input-permutation orbits: 3,984
- Classes after adding all candidates: 3,958

## Best ceiling gains

| feature_addition     |   descriptor_classes | language   |   ceiling_r2 |   ceiling_gain |
|:---------------------|---------------------:|:-----------|-------------:|---------------:|
| all_candidates       |                 3958 | AND_OR_NOT |       0.9999 |         0.0693 |
| anf_degree_profile   |                 3794 | AND_OR_NOT |       0.9947 |         0.0641 |
| fourier_signed_shape |                 3702 | AND_OR_NOT |       0.9903 |         0.0597 |
| all_candidates       |                 3958 | NAND       |       0.9996 |         0.1062 |
| anf_degree_profile   |                 3794 | NAND       |       0.9943 |         0.1009 |
| fourier_signed_shape |                 3702 | NAND       |       0.9913 |         0.0979 |
| all_candidates       |                 3958 | NOR        |       0.9996 |         0.1061 |
| anf_degree_profile   |                 3794 | NOR        |       0.9929 |         0.0995 |
| fourier_signed_shape |                 3702 | NOR        |       0.9913 |         0.0979 |

## Best single added coordinates

| feature_family       | component         |   descriptor_classes | language   |   ceiling_r2 |   ceiling_gain |
|:---------------------|:------------------|---------------------:|:-----------|-------------:|---------------:|
| fourier_sign_profile | degree_2_negative |                 1521 | AND_OR_NOT |       0.9556 |         0.0250 |
| fourier_sign_profile | degree_2_positive |                 1521 | AND_OR_NOT |       0.9556 |         0.0250 |
| anf_degree_profile   | degree_2          |                 1875 | AND_OR_NOT |       0.9527 |         0.0221 |
| fourier_sign_profile | degree_1_negative |                 1925 | AND_OR_NOT |       0.9507 |         0.0201 |
| fourier_sign_profile | degree_1_positive |                 1925 | AND_OR_NOT |       0.9507 |         0.0201 |
| anf_degree_profile   | degree_1          |                 1871 | NAND       |       0.9513 |         0.0578 |
| anf_degree_profile   | degree_2          |                 1875 | NAND       |       0.9334 |         0.0399 |
| fourier_sign_profile | degree_3_negative |                 1925 | NAND       |       0.9299 |         0.0365 |
| fourier_sign_profile | degree_3_positive |                 1925 | NAND       |       0.9299 |         0.0365 |
| fourier_sign_profile | degree_1_negative |                 1925 | NAND       |       0.9294 |         0.0360 |
| anf_degree_profile   | degree_1          |                 1871 | NOR        |       0.9382 |         0.0448 |
| anf_degree_profile   | degree_2          |                 1875 | NOR        |       0.9304 |         0.0370 |
| fourier_sign_profile | degree_3_negative |                 1925 | NOR        |       0.9299 |         0.0365 |
| fourier_sign_profile | degree_3_positive |                 1925 | NOR        |       0.9299 |         0.0365 |
| fourier_sign_profile | degree_1_negative |                 1925 | NOR        |       0.9294 |         0.0360 |

## Strict prediction with compact additions

| feature_addition                    |   AND_OR_NOT |   NAND |    NOR |
|:------------------------------------|-------------:|-------:|-------:|
| current_descriptor                  |       0.8435 | 0.7774 | 0.7774 |
| plus_anf_linear_count               |       0.8982 | 0.8568 | 0.8391 |
| plus_anf_quadratic_count            |       0.8870 | 0.8488 | 0.8404 |
| plus_certificate_profile            |       0.8075 | 0.8116 | 0.8116 |
| plus_fourier_degree2_negative_count |       0.8598 | 0.8500 | 0.8469 |
| plus_three_phase_counts             |       0.8471 | 0.8488 | 0.8044 |

## Largest exact-descriptor collisions

|   descriptor_class | language   |   spread |   low_complexity |   high_complexity |   low_truth_table |   high_truth_table | low_truth_table_hex   | high_truth_table_hex   | candidate_families_that_separate_pair                                                                                                   |
|-------------------:|:-----------|---------:|-----------------:|------------------:|------------------:|-------------------:|:----------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------|
|                597 | NAND       |        8 |                9 |                17 |             34687 |              61161 | 0x877f                | 0xeee9                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                102 | NOR        |        8 |                9 |                17 |               286 |              26760 | 0x011e                | 0x6888                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                 62 | NAND       |        7 |               15 |                22 |             26752 |                278 | 0x6880                | 0x0116                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                102 | NAND       |        7 |               12 |                19 |             26760 |                286 | 0x6888                | 0x011e                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                 62 | NOR        |        7 |               12 |                19 |               278 |              26752 | 0x0116                | 0x6880                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                597 | NOR        |        7 |               12 |                19 |             61161 |              34687 | 0xeee9                | 0x877f                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile                                                                            |
|                 84 | AND_OR_NOT |        5 |                7 |                12 |               598 |               1921 | 0x0256                | 0x0781                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile;local_sensitivity_profile;constant_restriction_profile;certificate_profile |
|                168 | AND_OR_NOT |        5 |                8 |                13 |             34951 |               6286 | 0x8887                | 0x188e                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile;local_sensitivity_profile;constant_restriction_profile;certificate_profile |
|                342 | AND_OR_NOT |        5 |                7 |                12 |             35768 |               6030 | 0x8bb8                | 0x178e                 | fourier_signed_shape;fourier_sign_profile;anf_degree_profile;constant_restriction_profile;certificate_profile                           |

## Interpretation

The strongest missing signal is algebraic support and Fourier sign structure,
not additional spectral magnitude: Fourier L1-by-degree barely changes the
ceiling, while ANF monomial counts and Fourier sign counts change it sharply.
The worst NAND collision makes this concrete: `0x877f` has complexity 9 and
`0xeee9` has complexity 17, while the latter is the former XOR
`a XOR b XOR c XOR d`. The current descriptor aliases the pair because it
retains Fourier energy but discards phase/sign information.

This survives strict prediction rather than only raising a ceiling. One ANF
coordinate--the number of linear monomials--raises strict R2 from .777 to
.857 for NAND and from .843 to .898 for AND/OR/NOT. Compact phase-sensitive
coordinates therefore recover real out-of-class signal. Naively combining
three such counts performs worse than the best single count, confirming that
metric design--not feature accumulation--is now part of the remaining gap.

The full ANF and signed-Fourier profiles nearly identify input-permutation
orbits, so their near-perfect ceilings are not themselves a compact theory.
The next target is a compositional measure of parity/XOR burden, beginning
with low-degree ANF support counts, followed by strict holdout at five inputs.
