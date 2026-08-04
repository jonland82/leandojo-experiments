# Semantic theorem/proof embeddings at 10,000 proofs

## Summary

The frozen selection rule chose statement **K=16**, proof **K=4**, joint **K=16**. Statement-only and proof-only clusters have AMI **0.0804** (Cramer's V **0.3372**). This is the direct semantic-view analogue of the earlier style/domain comparison.

No candidate in any view met the preregistered-style stability threshold of mean subsample AMI >= 0.8. The reported K values therefore come from the frozen fallback rule (maximum observed stability), and should be treated as exploratory resolutions rather than evidence for natural cluster counts.

The joint representation aligns with statement clusters at AMI **0.3623** and with proof clusters at AMI **0.1516**. These values indicate which side of the concatenated representation dominates its geometry.

The external comparisons provide a useful convergent-validity check: statement embeddings align more with the earlier domain topics (AMI **0.1652**) than style topics (**0.0319**), while proof embeddings align more with style (**0.2085**) than domain (**0.0602**).

These are empirical code/text embedding clusters, not equivalence classes in Lean's logic. The model was not trained to prove definitional or propositional equivalence.

## Isolation and provenance

This experiment is stored entirely in `experiments/semantic-embeddings-10000/`. It did not modify `experiments/aws-10000/`, `FINDINGS.md`, `out/`, or `app/data.js`. The earlier artifact was read only for exact sample-alignment validation and comparison labels.

The sample reproduces the seed-0 uniform sample of 10,000 records from 52,187 records with nonempty tactic traces. Exact order alignment with the earlier artifact: **true**.

The theorem declaration text came from `corpus.jsonl`. Lookup used theorem full name plus a normalized source-path suffix match; this disambiguates names appearing in more than one corpus file.

| Split | Available | Selected |
|---|---:|---:|
| train.json | 50,247 | 9,637 |
| val.json | 979 | 185 |
| test.json | 961 | 178 |

## Representations and embedding run

| View | Records | Characters | Vector shape | SHA-256 |
|---|---:|---:|---:|---|
| statement | 10,000 | 1,496,751 | 10,000 x 1,024 | `930e01dc5d22debc…` |
| proof | 10,000 | 2,266,862 | 10,000 x 1,024 | `c58b01459d4fa178…` |
| joint | 10,000 | 3,783,613 | 10,000 x 1,024 | `7d3cefa4b6ab526b…` |

Embeddings were generated with `cohere.embed-v4:0` through the AWS CLI command `aws bedrock-runtime invoke-model` in `us-east-1`, using `input_type=clustering`, 1024 float dimensions, and no truncation. Inputs were batched in source order and each completed batch was checkpointed before assembly.

Embedding wall time was **17.6 minutes**. Bedrock did not return a billing-token total, so cost is estimated from characters: **$0.23–$0.36** at the recorded $0.12/M-token rate.

## Clustering protocol

Each matrix was L2-normalized and clustered independently with MiniBatchKMeans. Candidate resolutions were K = 4, 6, 8, 10, 12, 14, and 16. For each K, a full-data fit was compared against four independent 80% subsample fits using adjusted mutual information. The frozen selection rule chooses the highest cosine silhouette among candidates with mean stability AMI at least 0.8; if none qualify, it chooses maximum stability.

### Candidate diagnostics

#### Statement

| K | Cosine silhouette | Mean subsample AMI | Min subsample AMI | Inertia / record | Selected |
|---:|---:|---:|---:|---:|:---:|
| 4 | 0.0429 | 0.3266 | 0.2544 | 0.4380 |  |
| 6 | 0.0287 | 0.2791 | 0.2211 | 0.4320 |  |
| 8 | 0.0281 | 0.3742 | 0.3294 | 0.4263 |  |
| 10 | 0.0322 | 0.3505 | 0.3230 | 0.4201 |  |
| 12 | 0.0283 | 0.4104 | 0.3816 | 0.4160 |  |
| 14 | 0.0274 | 0.3940 | 0.3727 | 0.4130 |  |
| 16 | 0.0314 | 0.4162 | 0.4028 | 0.4105 | yes |

#### Proof

| K | Cosine silhouette | Mean subsample AMI | Min subsample AMI | Inertia / record | Selected |
|---:|---:|---:|---:|---:|:---:|
| 4 | 0.0761 | 0.6224 | 0.5003 | 0.3881 | yes |
| 6 | 0.0443 | 0.5036 | 0.4562 | 0.3770 |  |
| 8 | 0.0690 | 0.4520 | 0.4182 | 0.3754 |  |
| 10 | 0.0641 | 0.4649 | 0.4362 | 0.3682 |  |
| 12 | 0.0614 | 0.5038 | 0.4977 | 0.3631 |  |
| 14 | 0.0444 | 0.5375 | 0.4632 | 0.3594 |  |
| 16 | 0.0332 | 0.5599 | 0.5271 | 0.3538 |  |

#### Joint

| K | Cosine silhouette | Mean subsample AMI | Min subsample AMI | Inertia / record | Selected |
|---:|---:|---:|---:|---:|:---:|
| 4 | 0.0300 | 0.3065 | 0.2837 | 0.4022 |  |
| 6 | 0.0329 | 0.4185 | 0.4053 | 0.3939 |  |
| 8 | 0.0319 | 0.4064 | 0.3821 | 0.3892 |  |
| 10 | 0.0338 | 0.4197 | 0.3908 | 0.3847 |  |
| 12 | 0.0323 | 0.4140 | 0.3725 | 0.3803 |  |
| 14 | 0.0325 | 0.4519 | 0.4240 | 0.3775 |  |
| 16 | 0.0306 | 0.4767 | 0.4537 | 0.3737 | yes |

## Cross-view alignment

| Comparison | N | AMI | NMI | Cramer's V |
|---|---:|---:|---:|---:|
| statement vs proof | 10,000 | 0.0804 | 0.0814 | 0.3372 |
| statement vs joint | 10,000 | 0.3623 | 0.3649 | 0.4749 |
| proof vs joint | 10,000 | 0.1516 | 0.1525 | 0.4736 |

## Alignment with the earlier feature views

| Semantic view vs earlier view | N | AMI | NMI | Cramer's V |
|---|---:|---:|---:|---:|
| statement vs style | 9,998 | 0.0319 | 0.0341 | 0.1506 |
| statement vs domain | 8,882 | 0.1652 | 0.1678 | 0.3697 |
| proof vs style | 9,998 | 0.2085 | 0.2090 | 0.4765 |
| proof vs domain | 8,882 | 0.0602 | 0.0610 | 0.2659 |
| joint vs style | 9,998 | 0.0631 | 0.0652 | 0.2125 |
| joint vs domain | 8,882 | 0.2090 | 0.2114 | 0.4410 |

## Alignment with source modules

| Semantic view | N | AMI | NMI | Cramer's V |
|---|---:|---:|---:|---:|
| statement | 10,000 | 0.1869 | 0.1944 | 0.3206 |
| proof | 10,000 | 0.0450 | 0.0475 | 0.2560 |
| joint | 10,000 | 0.2032 | 0.2105 | 0.3422 |

Statement and joint embeddings align much more strongly with the coarse source hierarchy than proof embeddings. This reproduces the earlier experiment's asymmetry: mathematical content is associated with where a theorem lives, while proof procedure travels more freely across modules.

## Selected cluster summaries

### Statement clusters (K=16)

| Cluster | Records | Share | Mean tactics | Leading source modules | Representative theorems |
|---:|---:|---:|---:|---|---|
| 0 | 349 | 3.5% | 5.03 | `Mathlib/RingTheory` (76), `Mathlib/Algebra` (61), `Mathlib/Data` (45) | `Ring.mul_inverse_cancel_left`, `Submodule.nontrivial_span_singleton`, `Polynomial.X_ne_zero` |
| 1 | 650 | 6.5% | 3.90 | `Mathlib/Data` (297), `Std/Data` (103), `Mathlib/Algebra` (58) | `Stream'.Seq.map_append`, `List.map_id'`, `Computation.map_think` |
| 2 | 788 | 7.9% | 3.21 | `Mathlib/Data` (200), `Mathlib/Algebra` (112), `Std/Data` (83) | `Int.neg_ne_of_pos`, `Std.Tactic.Omega.Int.add_le_iff_le_sub`, `Std.Tactic.Omega.Int.add_le_zero_iff_le_neg` |
| 3 | 925 | 9.2% | 4.26 | `Mathlib/CategoryTheory` (124), `Mathlib/Data` (123), `Mathlib/Topology` (106) | `List.map_surjective_iff`, `UpperSet.symm_map`, `LowerSet.symm_map` |
| 4 | 725 | 7.2% | 4.88 | `Mathlib/Data` (209), `Mathlib/RingTheory` (84), `Mathlib/Algebra` (78) | `npow_mul`, `npow_mul'`, `ppow_mul` |
| 5 | 869 | 8.7% | 2.31 | `Mathlib/Data` (259), `Mathlib/Analysis` (82), `Mathlib/Algebra` (78) | `unitInterval.symm_one`, `unitInterval.symm_zero`, `Ordinal.zero_nmul` |
| 6 | 542 | 5.4% | 3.94 | `Mathlib/Data` (124), `Mathlib/Analysis` (85), `Mathlib/Algebra` (74) | `NNReal.div_le_of_le_mul`, `Nat.div_mul_div_le_div`, `Nat.mul_div_le` |
| 7 | 735 | 7.3% | 3.82 | `Mathlib/Data` (224), `Mathlib/Topology` (180), `Mathlib/Order` (119) | `Finset.union_inter_cancel_left`, `iSup_iUnion`, `IsOpen.trans` |
| 8 | 766 | 7.7% | 3.78 | `Mathlib/Algebra` (173), `Mathlib/Data` (160), `Mathlib/LinearAlgebra` (107) | `Mathlib.Tactic.Ring.add_mul`, `TensorProduct.liftAux.smul`, `ArithmeticFunction.mul_smul'` |
| 9 | 587 | 5.9% | 5.11 | `Mathlib/Data` (117), `Mathlib/Topology` (112), `Mathlib/Analysis` (57) | `eVariationOn.mono`, `Finset.exists_mem_eq_sup'`, `Set.subset_image_iff` |
| 10 | 325 | 3.2% | 5.27 | `Mathlib/Analysis` (207), `Mathlib/Geometry` (46), `Mathlib/Topology` (44) | `hasFDerivWithinAt_inter`, `hasDerivAt_of_hasDerivAt_of_ne`, `DifferentiableWithinAt.hasFDerivWithinAt` |
| 11 | 639 | 6.4% | 3.40 | `Mathlib/CategoryTheory` (247), `Mathlib/Algebra` (50), `Mathlib/Topology` (50) | `CategoryTheory.Bicategory.leftUnitor_comp`, `CategoryTheory.Bicategory.leftUnitor_comp_inv`, `CategoryTheory.IsIso.Iso.inv_inv` |
| 12 | 579 | 5.8% | 5.93 | `Mathlib/MeasureTheory` (439), `Mathlib/Probability` (99), `Mathlib/Analysis` (25) | `MeasureTheory.lintegral_singleton'`, `MeasureTheory.set_lintegral_congr`, `MeasureTheory.lintegral_map'` |
| 13 | 431 | 4.3% | 6.63 | `Mathlib/Analysis` (222), `Mathlib/Topology` (73), `Mathlib/MeasureTheory` (65) | `lp.norm_le_of_forall_le'`, `tendsto_norm'`, `MeasureTheory.lintegral_ofReal_le_lintegral_nnnorm` |
| 14 | 600 | 6.0% | 4.06 | `Mathlib/Data` (219), `Std/Data` (59), `Mathlib/LinearAlgebra` (51) | `Fin.succ_inj`, `Finset.mem_fin`, `Finset.card_insertNone` |
| 15 | 490 | 4.9% | 4.42 | `Mathlib/Analysis` (78), `Mathlib/Algebra` (73), `Mathlib/LinearAlgebra` (71) | `QuadraticForm.map_sub`, `midpoint_sub_add`, `Submodule.ker_inclusion` |

### Proof clusters (K=4)

| Cluster | Records | Share | Mean tactics | Leading source modules | Representative theorems |
|---:|---:|---:|---:|---|---|
| 0 | 2,494 | 24.9% | 1.43 | `Mathlib/Data` (624), `Mathlib/Algebra` (259), `Mathlib/Analysis` (217) | `Cardinal.mk_list_eq_mk`, `Function.Antiperiodic.mul`, `Polynomial.map_natDegree_eq_sub` |
| 1 | 2,216 | 22.2% | 5.71 | `Mathlib/Data` (424), `Mathlib/Analysis` (329), `Mathlib/Algebra` (268) | `Commute.mul_self_sub_mul_self_eq'`, `mul_eq_mul_of_div_eq_div`, `inv_mul_le_iff_le_mul'` |
| 2 | 2,844 | 28.4% | 7.45 | `Mathlib/Data` (403), `Mathlib/Topology` (397), `Mathlib/MeasureTheory` (394) | `MeasureTheory.lintegral_iSup_ae`, `Set.PairwiseDisjoint.prod_left`, `Set.update_preimage_pi` |
| 3 | 2,446 | 24.5% | 1.84 | `Mathlib/Data` (616), `Mathlib/Algebra` (270), `Mathlib/Analysis` (227) | `Int.cast_nonpos`, `sub_one_mul`, `one_add_mul` |

### Joint clusters (K=16)

| Cluster | Records | Share | Mean tactics | Leading source modules | Representative theorems |
|---:|---:|---:|---:|---|---|
| 0 | 484 | 4.8% | 4.75 | `Mathlib/Data` (239), `Std/Data` (145), `Mathlib/Combinatorics` (20) | `List.perm_replicate`, `List.nth_le_tails`, `List.take_left'` |
| 1 | 603 | 6.0% | 6.52 | `Mathlib/RingTheory` (155), `Mathlib/LinearAlgebra` (146), `Mathlib/Algebra` (73) | `IsAssociatedPrime.map_of_injective`, `Submodule.exists_le_ker_of_lt_top`, `FractionalIdeal.mem_spanSingleton` |
| 2 | 464 | 4.6% | 2.62 | `Mathlib/Data` (98), `Mathlib/Algebra` (90), `Mathlib/Analysis` (78) | `smul_neg`, `Int.ModEq.neg`, `QuadraticForm.map_sub` |
| 3 | 689 | 6.9% | 6.05 | `Mathlib/Data` (183), `Mathlib/Analysis` (122), `Mathlib/RingTheory` (99) | `Polynomial.coeff_mul_X_pow`, `npow_mul`, `Polynomial.coeff_mul_X_pow'` |
| 4 | 635 | 6.3% | 4.70 | `Mathlib/Data` (166), `Mathlib/SetTheory` (96), `Mathlib/Analysis` (73) | `Bool.le_of_lt`, `Cardinal.mul_lt_of_lt`, `Bool.lt_of_le_of_lt` |
| 5 | 764 | 7.6% | 3.13 | `Mathlib/Data` (169), `Mathlib/Order` (91), `Mathlib/Analysis` (61) | `eq_iff_eq_cancel_right`, `Eq.congr_right`, `CategoryTheory.Iso.symm_symm_eq` |
| 6 | 331 | 3.3% | 5.87 | `Mathlib/CategoryTheory` (188), `Mathlib/Topology` (45), `Mathlib/AlgebraicGeometry` (35) | `CategoryTheory.IsUniversalColimit.of_iso`, `AlgebraicGeometry.PresheafedSpace.restrictStalkIso_inv_eq_ofRestrict`, `TopCat.Presheaf.toPushforwardOfIso_app` |
| 7 | 509 | 5.1% | 5.65 | `Mathlib/Analysis` (279), `Mathlib/Topology` (138), `Mathlib/Geometry` (55) | `ContinuousLinearMap.dist_le_opNorm`, `lipschitzWith_iff_dist_le_mul`, `LipschitzWith.mul_edist_le` |
| 8 | 792 | 7.9% | 5.06 | `Mathlib/Topology` (265), `Mathlib/Data` (112), `Mathlib/MeasureTheory` (96) | `nhdsWithin_inter_of_mem'`, `IsCompact.compl_mem_sets_of_nhdsWithin`, `Set.subset_image_iff` |
| 9 | 718 | 7.2% | 1.42 | `Mathlib/Data` (175), `Mathlib/Analysis` (90), `Mathlib/Algebra` (84) | `unitInterval.symm_one`, `unitInterval.symm_zero`, `Mathlib.Tactic.Ring.mul_zero` |
| 10 | 815 | 8.2% | 3.54 | `Mathlib/Data` (324), `Mathlib/Order` (99), `Mathlib/Algebra` (69) | `BoundedContinuousFunction.sum_apply`, `Finset.union_inter_cancel_left`, `Finset.iSup_insert` |
| 11 | 655 | 6.6% | 2.45 | `Mathlib/Data` (234), `Mathlib/Analysis` (69), `Mathlib/Algebra` (57) | `DirectSum.ofIntCast`, `Fin.one_eq_zero_iff`, `Num.add_zero` |
| 12 | 637 | 6.4% | 7.43 | `Mathlib/MeasureTheory` (416), `Mathlib/Probability` (84), `Mathlib/Analysis` (82) | `MeasureTheory.lintegral_iSup_ae`, `MeasureTheory.lintegral_mono_ae`, `MeasureTheory.set_lintegral_nnnorm_condexpIndSMul_le` |
| 13 | 551 | 5.5% | 3.08 | `Mathlib/Data` (98), `Mathlib/RingTheory` (81), `Mathlib/LinearAlgebra` (81) | `Submodule.map_id`, `TensorProduct.map_smul_left`, `LowerSet.mem_map` |
| 14 | 828 | 8.3% | 3.39 | `Mathlib/Algebra` (207), `Mathlib/Data` (174), `Mathlib/Analysis` (79) | `one_nsmul`, `mul_eq_of_eq_mul_inv`, `Units.inv_mul_cancel_left` |
| 15 | 525 | 5.2% | 2.62 | `Mathlib/CategoryTheory` (182), `Mathlib/Data` (60), `Mathlib/Algebra` (54) | `CategoryTheory.Bicategory.leftUnitor_comp`, `LieHom.comp_id`, `FreeAbelianGroup.map_comp` |

## Limitations

- Cohere Embed v4 is a general code/text embedding model, not a Lean kernel or a model of proof equivalence.
- Cluster labels describe geometry at the selected resolution; they are not a canonical taxonomy.
- The joint representation is a single embedding of concatenated fields, so it does not explicitly balance statement and proof information.
- Tactic proofs expose only the traced source tactics. Automation can invoke internal lemmas not listed in tactic syntax.
- Silhouette values in high-dimensional semantic spaces are often small; stability and external alignment should be considered alongside them.

## Reproduction

From the repository root:

```powershell
python experiments/semantic-embeddings-10000/scripts/prepare_inputs.py
python experiments/semantic-embeddings-10000/scripts/embed_aws_cli.py
python experiments/semantic-embeddings-10000/scripts/analyze.py
```

The embedding step is resumable: existing valid batch arrays are reused. Exact settings are frozen in `config.json`; input and embedding checksums are recorded in `artifacts/`.

## Runtime environment

- Python: `3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]`
- NumPy: `2.1.3`
- SciPy: `1.15.3`
- scikit-learn: `1.6.1`
- Platform: `Windows-11-10.0.22631-SP0`
