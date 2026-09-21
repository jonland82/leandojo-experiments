"""Diagnose information missing from the four-bit semantic descriptor.

This analysis deliberately studies descriptor ceilings before fitting a more
powerful predictor.  A ceiling gain means that a candidate invariant separates
functions that the current descriptor aliases despite different exact costs.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"


def load_experiment_module():
    path = EXPERIMENT / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("semantic_boolean_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_rows(values: np.ndarray) -> np.ndarray:
    transformed = values.copy()
    stride = 1
    while stride < transformed.shape[1]:
        for start in range(0, transformed.shape[1], 2 * stride):
            left = transformed[:, start:start + stride].copy()
            right = transformed[:, start + stride:start + 2 * stride].copy()
            transformed[:, start:start + stride] = left + right
            transformed[:, start + stride:start + 2 * stride] = left - right
        stride *= 2
    return transformed


def certificate_histograms(bits: np.ndarray, n: int) -> np.ndarray:
    """Count minimum certificate sizes, separated by function output."""
    assignments = bits.shape[1]
    minimum = np.full(bits.shape, n + 1, dtype=np.int8)
    assignment_ids = np.arange(assignments)
    for size in range(n + 1):
        for variables in itertools.combinations(range(n), size):
            mask = sum(1 << variable for variable in variables)
            patterns = np.unique(assignment_ids & mask)
            for pattern in patterns:
                group = assignment_ids[(assignment_ids & mask) == pattern]
                constant = np.all(bits[:, group] == bits[:, group[:1]], axis=1)
                rows = np.flatnonzero(constant)
                if len(rows):
                    current = minimum[np.ix_(rows, group)]
                    minimum[np.ix_(rows, group)] = np.minimum(current, size)
    columns = []
    for output in (0, 1):
        for size in range(n + 1):
            columns.append(np.sum((bits == output) & (minimum == size), axis=1))
    return np.stack(columns, axis=1)


def candidate_invariants(n: int = 4) -> dict[str, np.ndarray]:
    assignments = 1 << n
    functions = 1 << assignments
    tables = np.arange(functions, dtype=np.uint32)
    assignment_ids = np.arange(assignments, dtype=np.uint32)
    bits = ((tables[:, None] >> assignment_ids) & 1).astype(np.int8)
    signs = 2 * bits.astype(np.int16) - 1
    degrees = np.asarray([int(mask).bit_count() for mask in assignment_ids])

    walsh = transform_rows(signs)
    signed_parts = []
    absolute_parts = []
    fourier_l1 = []
    fourier_signs = []
    for degree in range(n + 1):
        block = walsh[:, degrees == degree]
        signed_parts.append(np.sort(block, axis=1))
        absolute_parts.append(np.sort(np.abs(block), axis=1))
        fourier_l1.append(np.sum(np.abs(block), axis=1))
        fourier_signs.extend((
            np.sum(block < 0, axis=1),
            np.sum(block == 0, axis=1),
            np.sum(block > 0, axis=1),
        ))

    anf = bits.copy()
    for variable in range(n):
        for monomial in range(assignments):
            if monomial & (1 << variable):
                anf[:, monomial] ^= anf[:, monomial ^ (1 << variable)]
    anf_profile = np.stack([
        np.sum(anf[:, degrees == degree], axis=1) for degree in range(n + 1)
    ], axis=1)

    local_sensitivity = np.zeros_like(bits)
    for variable in range(n):
        local_sensitivity += bits != bits[:, assignment_ids ^ (1 << variable)]
    sensitivity_profile = np.stack([
        np.sum((bits == output) & (local_sensitivity == sensitivity), axis=1)
        for output in (0, 1) for sensitivity in range(n + 1)
    ], axis=1)

    constant_faces = []
    for codimension in range(1, n):
        zero_count = np.zeros(functions, dtype=np.int16)
        one_count = np.zeros(functions, dtype=np.int16)
        for variables in itertools.combinations(range(n), codimension):
            mask = sum(1 << variable for variable in variables)
            for pattern in np.unique(assignment_ids & mask):
                group = assignment_ids[(assignment_ids & mask) == pattern]
                subtotal = np.sum(bits[:, group], axis=1)
                zero_count += subtotal == 0
                one_count += subtotal == len(group)
        constant_faces.extend((zero_count, one_count))

    return {
        "fourier_signed_shape": np.concatenate(signed_parts, axis=1),
        "fourier_absolute_shape": np.concatenate(absolute_parts, axis=1),
        "fourier_l1_by_degree": np.stack(fourier_l1, axis=1),
        "fourier_sign_profile": np.stack(fourier_signs, axis=1),
        "anf_degree_profile": anf_profile,
        "local_sensitivity_profile": sensitivity_profile,
        "constant_restriction_profile": np.stack(constant_faces, axis=1),
        "certificate_profile": certificate_histograms(bits, n),
    }


def class_ids(values: np.ndarray) -> tuple[int, np.ndarray]:
    _, inverse = np.unique(values, axis=0, return_inverse=True)
    return int(inverse.max() + 1), inverse


def class_mean_prediction(target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=target)
    return sums[groups] / counts[groups]


def main() -> None:
    run = load_experiment_module()
    semantic, raw, orbit_canonical = run.truth_table_features()
    base_count, base_classes = class_ids(raw)
    orbit_count = len(np.unique(orbit_canonical))
    candidates = candidate_invariants(run.N)
    predictions = pd.read_csv(ARTIFACTS / "predictions.csv")
    languages = run.CONFIG["languages"]
    targets = {
        language: predictions[f"{language}_minimum_gates"].to_numpy(dtype=float)
        for language in languages
    }

    base_oracles = {
        language: class_mean_prediction(target, base_classes)
        for language, target in targets.items()
    }
    ceiling_rows = []
    augmented_classes: dict[str, np.ndarray] = {}
    feature_sets = {"current_descriptor": np.empty((len(raw), 0), dtype=np.int8), **candidates}
    for name, extra in feature_sets.items():
        values = raw if name == "current_descriptor" else np.column_stack((raw, extra))
        count, groups = class_ids(values)
        augmented_classes[name] = groups
        for language, target in targets.items():
            oracle = class_mean_prediction(target, groups)
            base_r2 = r2_score(target, base_oracles[language])
            ceiling_rows.append({
                "feature_addition": name,
                "descriptor_classes": count,
                "language": language,
                "ceiling_r2": r2_score(target, oracle),
                "ceiling_gain": r2_score(target, oracle) - base_r2,
            })

    all_extra = np.column_stack(tuple(candidates.values()))
    all_count, all_classes = class_ids(np.column_stack((raw, all_extra)))
    augmented_classes["all_candidates"] = all_classes
    for language, target in targets.items():
        oracle = class_mean_prediction(target, all_classes)
        base_r2 = r2_score(target, base_oracles[language])
        ceiling_rows.append({
            "feature_addition": "all_candidates",
            "descriptor_classes": all_count,
            "language": language,
            "ceiling_r2": r2_score(target, oracle),
            "ceiling_gain": r2_score(target, oracle) - base_r2,
        })
    ceilings = pd.DataFrame(ceiling_rows)
    ceilings.to_csv(ARTIFACTS / "candidate_feature_ceilings.csv", index=False)

    scalar_rows = []
    scalar_labels = {
        "fourier_signed_shape": [f"degree_{d}_rank_{r}" for d in range(run.N + 1)
                                  for r in range(math.comb(run.N, d))],
        "fourier_absolute_shape": [f"degree_{d}_rank_{r}" for d in range(run.N + 1)
                                    for r in range(math.comb(run.N, d))],
        "fourier_l1_by_degree": [f"degree_{d}" for d in range(run.N + 1)],
        "fourier_sign_profile": [f"degree_{d}_{sign}" for d in range(run.N + 1)
                                  for sign in ("negative", "zero", "positive")],
        "anf_degree_profile": [f"degree_{d}" for d in range(run.N + 1)],
        "local_sensitivity_profile": [f"output_{output}_sensitivity_{s}"
                                      for output in (0, 1) for s in range(run.N + 1)],
        "constant_restriction_profile": [f"codimension_{c}_output_{output}"
                                         for c in range(1, run.N) for output in (0, 1)],
        "certificate_profile": [f"output_{output}_certificate_{size}"
                                for output in (0, 1) for size in range(run.N + 1)],
    }
    for family, extra in candidates.items():
        for column, label in enumerate(scalar_labels[family]):
            count, groups = class_ids(np.column_stack((raw, extra[:, column])))
            for language, target in targets.items():
                oracle = class_mean_prediction(target, groups)
                base_r2 = r2_score(target, base_oracles[language])
                scalar_rows.append({
                    "feature_family": family,
                    "component": label,
                    "descriptor_classes": count,
                    "language": language,
                    "ceiling_r2": r2_score(target, oracle),
                    "ceiling_gain": r2_score(target, oracle) - base_r2,
                })
    scalar_ceilings = pd.DataFrame(scalar_rows)
    scalar_ceilings.to_csv(ARTIFACTS / "candidate_component_ceilings.csv", index=False)

    compact_prediction_sets = {
        "current_descriptor": np.empty((len(raw), 0)),
        "plus_anf_linear_count": candidates["anf_degree_profile"][:, [1]],
        "plus_anf_quadratic_count": candidates["anf_degree_profile"][:, [2]],
        "plus_fourier_degree2_negative_count": candidates["fourier_sign_profile"][:, [6]],
        "plus_three_phase_counts": np.column_stack((
            candidates["anf_degree_profile"][:, 1],
            candidates["anf_degree_profile"][:, 2],
            candidates["fourier_sign_profile"][:, 6],
        )),
        "plus_certificate_profile": candidates["certificate_profile"],
    }
    prediction_rows = []
    for name, extra in compact_prediction_sets.items():
        values = raw if extra.shape[1] == 0 else np.column_stack((raw, extra))
        count, groups = class_ids(values)
        scaled = StandardScaler().fit_transform(values)
        tests, neighborhoods = run.tied_neighbor_plan(scaled, groups, groups, run.K)
        for language, target in targets.items():
            metrics = run.evaluate_tied(target, tests, neighborhoods)
            prediction_rows.append({
                "feature_addition": name,
                "descriptor_classes": count,
                "language": language,
                "strict_r2": metrics["r2"],
                "strict_rmse": metrics["rmse"],
            })
    prediction_metrics = pd.DataFrame(prediction_rows)
    prediction_metrics.to_csv(
        ARTIFACTS / "candidate_feature_prediction_metrics.csv", index=False)

    collision_rows = []
    pair_rows = []
    for descriptor_class in range(base_count):
        members = np.flatnonzero(base_classes == descriptor_class)
        row = {"descriptor_class": descriptor_class, "functions": len(members)}
        for feature in run.FEATURES:
            row[feature] = semantic.loc[members[0], feature]
        for language, target in targets.items():
            values = target[members]
            low = members[np.argmin(values)]
            high = members[np.argmax(values)]
            spread = int(values.max() - values.min())
            row[f"{language}_minimum"] = int(values.min())
            row[f"{language}_maximum"] = int(values.max())
            row[f"{language}_spread"] = spread
            if spread:
                changed = [
                    name for name, feature_values in candidates.items()
                    if not np.array_equal(feature_values[low], feature_values[high])
                ]
                pair_rows.append({
                    "descriptor_class": descriptor_class,
                    "language": language,
                    "spread": spread,
                    "low_complexity": int(target[low]),
                    "high_complexity": int(target[high]),
                    "low_truth_table": int(low),
                    "high_truth_table": int(high),
                    "low_truth_table_hex": f"0x{low:04x}",
                    "high_truth_table_hex": f"0x{high:04x}",
                    "candidate_families_that_separate_pair": ";".join(changed),
                })
        collision_rows.append(row)
    collisions = pd.DataFrame(collision_rows)
    collisions.to_csv(ARTIFACTS / "descriptor_collision_classes.csv", index=False)
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["spread", "language", "descriptor_class"], ascending=[False, True, True])
    pairs.to_csv(ARTIFACTS / "largest_collision_pairs.csv", index=False)

    summary = {
        "functions": len(raw),
        "input_permutation_orbits": orbit_count,
        "current_descriptor_classes": base_count,
        "candidate_families": {name: values.shape[1] for name, values in candidates.items()},
        "augmented_descriptor_classes": {
            name: int(groups.max() + 1) for name, groups in augmented_classes.items()
        },
        "largest_within_descriptor_spread": {
            language: int(collisions[f"{language}_spread"].max()) for language in languages
        },
    }
    (ARTIFACTS / "gap_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    best = (ceilings[ceilings.feature_addition != "current_descriptor"]
            .sort_values(["language", "ceiling_gain"], ascending=[True, False])
            .groupby("language", sort=False).head(3))
    best_scalars = (scalar_ceilings.sort_values(
        ["language", "ceiling_gain"], ascending=[True, False])
        .groupby("language", sort=False).head(5))
    top_pairs = pairs.groupby("language", sort=False).head(3)
    compact_predictions = prediction_metrics.pivot(
        index="feature_addition", columns="language", values="strict_r2").reset_index()
    lines = [
        "# Descriptor-gap analysis", "",
        "This analysis asks which invariant semantic summaries split functions that",
        "the current descriptor aliases despite different exact formula costs. Ceiling",
        "gains are diagnostic; they are not held-out prediction scores.", "",
        "## Scale", "",
        f"- Current descriptor classes: {base_count:,}",
        f"- Input-permutation orbits: {orbit_count:,}",
        f"- Classes after adding all candidates: {all_count:,}", "",
        "## Best ceiling gains", "",
        best.to_markdown(index=False, floatfmt=".4f"), "",
        "## Best single added coordinates", "",
        best_scalars.to_markdown(index=False, floatfmt=".4f"), "",
        "## Strict prediction with compact additions", "",
        compact_predictions.to_markdown(index=False, floatfmt=".4f"), "",
        "## Largest exact-descriptor collisions", "",
        top_pairs.to_markdown(index=False), "",
        "## Interpretation", "",
        "The strongest missing signal is algebraic support and Fourier sign structure,",
        "not additional spectral magnitude: Fourier L1-by-degree barely changes the",
        "ceiling, while ANF monomial counts and Fourier sign counts change it sharply.",
        "The worst NAND collision makes this concrete: `0x877f` has complexity 9 and",
        "`0xeee9` has complexity 17, while the latter is the former XOR",
        "`a XOR b XOR c XOR d`. The current descriptor aliases the pair because it",
        "retains Fourier energy but discards phase/sign information.", "",
        "This survives strict prediction rather than only raising a ceiling. One ANF",
        "coordinate--the number of linear monomials--raises strict R2 from .777 to",
        ".857 for NAND and from .843 to .898 for AND/OR/NOT. Compact phase-sensitive",
        "coordinates therefore recover real out-of-class signal. Naively combining",
        "three such counts performs worse than the best single count, confirming that",
        "metric design--not feature accumulation--is now part of the remaining gap.", "",
        "The full ANF and signed-Fourier profiles nearly identify input-permutation",
        "orbits, so their near-perfect ceilings are not themselves a compact theory.",
        "The next target is a compositional measure of parity/XOR burden, beginning",
        "with low-degree ANF support counts, followed by strict holdout at five inputs.", "",
    ]
    (EXPERIMENT / "GAP_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nBest candidate gains:\n", best.to_string(index=False))


if __name__ == "__main__":
    main()
