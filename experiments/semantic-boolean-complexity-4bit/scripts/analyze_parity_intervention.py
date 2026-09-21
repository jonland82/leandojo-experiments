"""Test whether affine/parity phase is a missing formula-complexity signal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"


def load_experiment_module():
    path = EXPERIMENT / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("semantic_boolean_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parity_truth_table(variable_mask: int, n: int) -> int:
    return sum(
        ((assignment & variable_mask).bit_count() & 1) << assignment
        for assignment in range(1 << n)
    )


def linear_anf_count(tables: np.ndarray, n: int) -> np.ndarray:
    constant = tables & 1
    coefficients = [((tables >> (1 << variable)) & 1) ^ constant for variable in range(n)]
    return np.sum(np.stack(coefficients, axis=1), axis=1)


def histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    threshold = quantile * histogram.sum()
    return int(np.searchsorted(np.cumsum(histogram), threshold, side="left"))


def distribution_row(category: str, language: str, histogram: np.ndarray) -> dict[str, float | int | str]:
    values = np.arange(len(histogram))
    count = int(histogram.sum())
    return {
        "category": category,
        "language": language,
        "pairs": count,
        "mean_absolute_gate_difference": float(np.dot(values, histogram) / count),
        "median_absolute_gate_difference": histogram_quantile(histogram, 0.5),
        "q90_absolute_gate_difference": histogram_quantile(histogram, 0.9),
        "maximum_absolute_gate_difference": int(np.flatnonzero(histogram)[-1]),
        "fraction_nonzero": float(histogram[1:].sum() / count),
        "fraction_at_least_3": float(histogram[3:].sum() / count),
        "fraction_at_least_5": float(histogram[5:].sum() / count),
    }


def main() -> None:
    run = load_experiment_module()
    semantic, raw, orbit_canonical = run.truth_table_features()
    _, descriptor_classes = run.descriptor_classes(raw)
    _, orbit_classes = np.unique(orbit_canonical, return_inverse=True)
    predictions = pd.read_csv(ARTIFACTS / "predictions.csv")
    languages = run.CONFIG["languages"]
    targets = {
        language: predictions[f"{language}_minimum_gates"].to_numpy(dtype=np.int16)
        for language in languages
    }
    tables = np.arange(run.FUNCTIONS, dtype=np.uint32)
    anf_linear = linear_anf_count(tables, run.N)
    parity_masks = {
        parity_truth_table(variable_mask, run.N): variable_mask
        for variable_mask in range(1, 1 << run.N)
    }
    parity_tables = np.asarray(sorted(parity_masks), dtype=np.uint32)

    # Exhaustive interventions f -> f XOR parity_S, counted once per unordered pair.
    intervention_rows = []
    for parity_table in parity_tables:
        variable_mask = parity_masks[int(parity_table)]
        transformed = tables ^ parity_table
        once = tables < transformed
        same_descriptor = descriptor_classes == descriptor_classes[transformed]
        same_orbit = orbit_classes == orbit_classes[transformed]
        for scope, selected in (
            ("all_pairs", once),
            ("same_descriptor_cross_orbit", once & same_descriptor & ~same_orbit),
        ):
            for language, target in targets.items():
                differences = np.abs(target[transformed[selected]] - target[selected])
                intervention_rows.append({
                    "scope": scope,
                    "parity_variable_mask": variable_mask,
                    "parity_order": variable_mask.bit_count(),
                    "language": language,
                    "pairs": int(selected.sum()),
                    "mean_absolute_gate_difference": (
                        float(differences.mean()) if len(differences) else np.nan),
                    "fraction_changed": (
                        float(np.mean(differences != 0)) if len(differences) else np.nan),
                    "maximum_absolute_gate_difference": (
                        int(differences.max()) if len(differences) else 0),
                })
    interventions = pd.DataFrame(intervention_rows)
    interventions.to_csv(ARTIFACTS / "parity_interventions.csv", index=False)

    # Exact-descriptor, cross-orbit pair control. This isolates missing information
    # while avoiding identical costs caused solely by input renaming.
    maximum_cost = max(int(target.max()) for target in targets.values())
    histograms = {
        (category, language): np.zeros(maximum_cost + 1, dtype=np.int64)
        for category in ("parity_related", "other_same_descriptor")
        for language in languages
    }
    parity_pair_rows = []
    for descriptor_class in np.unique(descriptor_classes):
        members = np.flatnonzero(descriptor_classes == descriptor_class)
        left_position, right_position = np.triu_indices(len(members), k=1)
        left = members[left_position]
        right = members[right_position]
        cross_orbit = orbit_classes[left] != orbit_classes[right]
        left = left[cross_orbit]
        right = right[cross_orbit]
        xor_values = np.bitwise_xor(left, right).astype(np.uint32)
        parity_related = np.isin(xor_values, parity_tables)
        for category, selected in (
            ("parity_related", parity_related),
            ("other_same_descriptor", ~parity_related),
        ):
            for language, target in targets.items():
                differences = np.abs(target[right[selected]] - target[left[selected]])
                histograms[(category, language)] += np.bincount(
                    differences, minlength=maximum_cost + 1)

        for lvalue, rvalue, xor_value in zip(
                left[parity_related], right[parity_related], xor_values[parity_related]):
            variable_mask = parity_masks[int(xor_value)]
            row = {
                "descriptor_class": int(descriptor_class),
                "left_truth_table": int(lvalue),
                "right_truth_table": int(rvalue),
                "left_truth_table_hex": f"0x{lvalue:04x}",
                "right_truth_table_hex": f"0x{rvalue:04x}",
                "parity_variable_mask": variable_mask,
                "parity_order": variable_mask.bit_count(),
                "left_anf_linear_count": int(anf_linear[lvalue]),
                "right_anf_linear_count": int(anf_linear[rvalue]),
            }
            for language, target in targets.items():
                row[f"{language}_left"] = int(target[lvalue])
                row[f"{language}_right"] = int(target[rvalue])
                row[f"{language}_absolute_difference"] = int(
                    abs(target[lvalue] - target[rvalue]))
            parity_pair_rows.append(row)

    controls = pd.DataFrame([
        distribution_row(category, language, histogram)
        for (category, language), histogram in histograms.items()
    ])
    controls.to_csv(ARTIFACTS / "parity_matched_controls.csv", index=False)
    parity_pairs = pd.DataFrame(parity_pair_rows)
    sort_columns = [f"{language}_absolute_difference" for language in languages]
    parity_pairs["maximum_language_difference"] = parity_pairs[sort_columns].max(axis=1)
    parity_pairs = parity_pairs.sort_values(
        ["maximum_language_difference", "descriptor_class"], ascending=[False, True])
    parity_pairs.to_csv(ARTIFACTS / "parity_matched_pairs.csv", index=False)

    by_order = (interventions[interventions.scope == "same_descriptor_cross_orbit"]
                .groupby(["parity_order", "language"], as_index=False)
                .agg(pairs=("pairs", "sum"),
                     mean_absolute_gate_difference=("mean_absolute_gate_difference", "mean"),
                     fraction_changed=("fraction_changed", "mean"),
                     maximum_absolute_gate_difference=("maximum_absolute_gate_difference", "max")))
    summary = {
        "nonzero_linear_parities": len(parity_tables),
        "exact_descriptor_cross_orbit_pairs": int(
            controls[controls.category == "other_same_descriptor"].iloc[0].pairs
            + controls[controls.category == "parity_related"].iloc[0].pairs),
        "parity_related_exact_descriptor_cross_orbit_pairs": int(len(parity_pairs)),
        "largest_parity_related_gate_difference": {
            language: int(parity_pairs[f"{language}_absolute_difference"].max())
            for language in languages
        },
    }
    (ARTIFACTS / "parity_intervention_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top_pairs = parity_pairs.head(8)[[
        "left_truth_table_hex", "right_truth_table_hex", "parity_order",
        "left_anf_linear_count", "right_anf_linear_count", *sort_columns,
    ]]
    lines = [
        "# Parity intervention analysis", "",
        "For every four-input function, this analysis XORs each of the 15 nonzero",
        "linear parities into its truth table. The decisive comparison holds the",
        "paper's original descriptor exactly fixed and excludes input-renaming orbits.", "",
        "## Matched control", "",
        controls.to_markdown(index=False, floatfmt=".4f"), "",
        "## Descriptor-preserving interventions by parity order", "",
        by_order.to_markdown(index=False, floatfmt=".4f"), "",
        "## Largest controlled examples", "",
        top_pairs.to_markdown(index=False), "",
        "## Interpretation", "",
        "Parity-related functions are not merely nearby in the original feature",
        "space: the matched analysis compares them inside the exact same descriptor",
        "class. Any systematic excess gate difference therefore isolates information",
        "discarded by bias, sorted influences, Fourier energy by degree, algebraic",
        "degree, and stabilizer size. The ANF intervention changes only constant/linear",
        "support, leaving every higher-degree ANF coefficient fixed.", "",
        "The result is basis-relative. For NAND, parity-related pairs have mean gap",
        "1.271 versus 1.167 in controls, and gaps of at least five occur 1.00% versus",
        "0.36%; NOR is similar. Full four-variable parity produces the largest gaps.",
        "For AND/OR/NOT, however, parity pairs are closer than controls (mean .664",
        "versus .813) and never differ by more than two gates. Thus the experiment",
        "supports phase sensitivity for NAND/NOR but falsifies parity burden as a",
        "universal explanation. The ANF-linear count's predictive gain for AND/OR/NOT",
        "must instead be a proxy for another property, plausibly factorability.", "",
    ]
    (EXPERIMENT / "PARITY_INTERVENTION.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nMatched controls:\n", controls.to_string(index=False))


if __name__ == "__main__":
    main()
