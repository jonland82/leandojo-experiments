"""Stress-test semantic prediction and sorting-trajectory geometry."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
BOOLEAN_ATLAS = ROOT / "experiments" / "creativity-boolean-atlas" / "artifacts" / "boolean_atlas.csv"
FEATURES = (["bias", "algebraic_degree", "input_symmetries"]
            + [f"influence_{i}" for i in range(3)]
            + [f"fourier_energy_degree_{i}" for i in range(4)])


def permute_truth_table(table: int, permutation: tuple[int, ...]) -> int:
    output = 0
    for assignment in range(8):
        mapped = sum(((assignment >> permutation[i]) & 1) << i for i in range(3))
        output |= ((table >> mapped) & 1) << assignment
    return output


def boolean_groups(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    permutations = list(itertools.permutations(range(3)))
    orbit = np.asarray([min(permute_truth_table(table, permutation)
                            for permutation in permutations)
                        for table in base["truth_table"].astype(int)])
    descriptors = [tuple(round(float(row[column]), 12) for column in FEATURES)
                   for _, row in base.iterrows()]
    descriptor_ids = {descriptor: index for index, descriptor in enumerate(sorted(set(descriptors)))}
    descriptor = np.asarray([descriptor_ids[value] for value in descriptors])
    return orbit, descriptor


def neighbor_plan(features: np.ndarray, groups: np.ndarray, neighbors: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tests = []
    nearest = []
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        distances = np.linalg.norm(features[test, None, :] - features[None, train, :], axis=2)
        selected = train[np.argsort(distances, axis=1)[:, :neighbors]]
        tests.append(test)
        nearest.append(selected)
    return tests, nearest


def evaluate_plan(target: np.ndarray, tests: list[np.ndarray], nearest: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    prediction = np.empty_like(target, dtype=float)
    baseline = np.empty_like(target, dtype=float)
    all_indices = np.arange(len(target))
    for test, selected in zip(tests, nearest):
        prediction[test] = target[selected].mean(axis=1)
        train_mask = np.ones(len(target), dtype=bool)
        train_mask[test] = False
        baseline[test] = target[all_indices[train_mask]].mean()
    return {"r2": float(r2_score(target, prediction)),
            "rmse": float(mean_squared_error(target, prediction) ** 0.5),
            "fold_mean_rmse": float(mean_squared_error(target, baseline) ** 0.5),
            "prediction": prediction}


def validate_boolean() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    atlas = pd.read_csv(BOOLEAN_ATLAS)
    base = atlas[atlas["language"] == "NAND"].sort_values("truth_table").reset_index(drop=True)
    raw_features = base[FEATURES].to_numpy(dtype=float)
    features = StandardScaler().fit_transform(raw_features)
    orbit, descriptor = boolean_groups(base)
    naive = np.arange(len(base))
    plans = {"leave_one_function_out": neighbor_plan(features, naive, CONFIG["boolean_neighbors"]),
             "leave_one_orbit_out": neighbor_plan(features, orbit, CONFIG["boolean_neighbors"]),
             "leave_one_descriptor_class_out": neighbor_plan(
                 features, descriptor, CONFIG["boolean_neighbors"])}
    rng = np.random.default_rng(CONFIG["seed"])
    stratum_values = [tuple(row) for row in base[["bias", "algebraic_degree",
                                                  "input_symmetries"]].to_numpy()]
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, value in enumerate(stratum_values):
        strata[value].append(index)

    rows = []
    null_rows = []
    result = {"functions": len(base), "input_permutation_orbits": int(len(np.unique(orbit))),
              "invariant_descriptor_classes": int(len(np.unique(descriptor))),
              "languages": {}}
    for language in ("NAND", "NOR", "AND_OR_NOT"):
        subset = atlas[atlas["language"] == language].sort_values("truth_table")
        target = subset["minimum_gates"].to_numpy(dtype=float)
        language_result = {}
        for protocol, (tests, nearest) in plans.items():
            evaluated = evaluate_plan(target, tests, nearest)
            language_result[protocol] = {key: value for key, value in evaluated.items()
                                         if key != "prediction"}
            rows.append({"language": language, "protocol": protocol,
                         **language_result[protocol]})

        tests, nearest = plans["leave_one_descriptor_class_out"]
        observed = language_result["leave_one_descriptor_class_out"]["r2"]
        null_values = []
        for repetition in range(CONFIG["null_repetitions"]):
            shuffled = target.copy()
            for indices in strata.values():
                shuffled[indices] = rng.permutation(shuffled[indices])
            null_r2 = evaluate_plan(shuffled, tests, nearest)["r2"]
            null_values.append(null_r2)
            null_rows.append({"language": language, "repetition": repetition,
                              "null_r2": null_r2})
        null_array = np.asarray(null_values)
        language_result["stratified_null"] = {
            "repetitions": CONFIG["null_repetitions"],
            "mean_r2": float(null_array.mean()),
            "q025_r2": float(np.quantile(null_array, 0.025)),
            "q975_r2": float(np.quantile(null_array, 0.975)),
            "upper_tail_pvalue": float((1 + np.sum(null_array >= observed))
                                        / (CONFIG["null_repetitions"] + 1))}
        result["languages"][language] = language_result
    return result, pd.DataFrame(rows), pd.DataFrame(null_rows)


def apply_comparator(values: tuple[int, ...], comparator: tuple[int, int]) -> tuple[int, ...]:
    left, right = comparator
    if values[left] <= values[right]:
        return values
    output = list(values)
    output[left], output[right] = output[right], output[left]
    return tuple(output)


def validate_sorting() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    n = CONFIG["sorting_n"]
    permutations = list(itertools.permutations(range(n)))
    permutation_index = {value: index for index, value in enumerate(permutations)}
    comparators = list(itertools.combinations(range(n), 2))
    primitives = [[permutation_index[apply_comparator(value, comparator)]
                   for value in permutations] for comparator in comparators]
    initial = tuple(range(len(permutations)))
    correct = tuple([permutation_index[tuple(range(n))]] * len(permutations))

    states = [initial]
    state_ids = {initial: 0}
    transitions: list[list[int]] = []
    frontier = {0}
    for _ in range(CONFIG["sorting_maximum_length"]):
        next_frontier = set()
        for state_id in sorted(frontier):
            while len(transitions) <= state_id:
                transitions.append([])
            if transitions[state_id]:
                continue
            state = states[state_id]
            for primitive in primitives:
                target = tuple(primitive[value] for value in state)
                if target not in state_ids:
                    state_ids[target] = len(states)
                    states.append(target)
                    next_frontier.add(state_ids[target])
                transitions[state_id].append(state_ids[target])
        frontier |= next_frontier
    # Complete transitions for states discovered on the final expansion.
    for state_id, state in enumerate(states):
        if state_id < len(transitions) and transitions[state_id]:
            continue
        while len(transitions) <= state_id:
            transitions.append([])
        transitions[state_id] = [state_ids.setdefault(
            tuple(primitive[value] for value in state), len(state_ids))
            for primitive in primitives]
        if any(target >= len(states) for target in transitions[state_id]):
            raise RuntimeError("sorting automaton was not closed")
    correct_id = state_ids[correct]

    path_counts = {length: Counter() for length in range(
        CONFIG["sorting_minimum_length"], CONFIG["sorting_maximum_length"] + 1)}

    def walk(depth: int, state_id: int, path: tuple[int, ...]) -> None:
        if depth >= CONFIG["sorting_minimum_length"] and state_id == correct_id:
            path_counts[depth][path] += 1
        if depth == CONFIG["sorting_maximum_length"]:
            return
        for target in transitions[state_id]:
            walk(depth + 1, target, path + (target,))

    walk(0, 0, (0,))
    expected = {5: 12, 6: 900, 7: 16800, 8: 204120}
    observed = {length: int(sum(counter.values())) for length, counter in path_counts.items()}
    if observed != expected:
        raise RuntimeError(f"sorting count invariant failed: {observed}")

    histograms = np.asarray([np.bincount(state, minlength=len(permutations))
                             for state in states], dtype=np.int16)
    wire_inverse_maps = []
    for wire_permutation in itertools.permutations(range(n)):
        old_to_new = []
        for output in permutations:
            transformed = tuple(output[wire_permutation[i]] for i in range(n))
            old_to_new.append(permutation_index[transformed])
        wire_inverse_maps.append(np.argsort(old_to_new))

    quotient_rows = []
    summary_rows = []
    for length, counter in path_counts.items():
        quotient: dict[tuple[int, ...], dict] = {}
        entropy_profiles = set()
        for path, multiplicity in counter.items():
            matrix = histograms[np.asarray(path)]
            entropies = []
            for histogram in matrix:
                probabilities = histogram[histogram > 0] / len(permutations)
                entropies.append(round(float(-np.sum(probabilities * np.log2(probabilities))), 12))
            entropy_profiles.add(tuple(entropies))
            candidates = []
            for inverse in wire_inverse_maps:
                oriented = matrix[:, inverse]
                candidates.append((tuple(int(value) for value in oriented.ravel()), inverse))
            key, inverse = min(candidates, key=lambda item: item[0])
            if key not in quotient:
                oriented = matrix[:, inverse].astype(float) / len(permutations)
                grid = np.linspace(0.0, 1.0, 9)
                source = np.linspace(0.0, 1.0, len(oriented))
                resampled = np.column_stack([
                    np.interp(grid, source, oriented[:, column])
                    for column in range(len(permutations))])
                quotient[key] = {"multiplicity": 0, "features": resampled.ravel(),
                                 "first_sorted_step": next(index for index, state_id in enumerate(path)
                                                           if state_id == correct_id)}
            quotient[key]["multiplicity"] += multiplicity
        for key, values in quotient.items():
            quotient_rows.append({"length": length,
                                  "multiplicity": values["multiplicity"],
                                  "padded": values["first_sorted_step"] < length,
                                  "features": values["features"]})
        summary_rows.append({"length": length,
                             "correct_sequences": observed[length],
                             "raw_full_state_trajectories": len(counter),
                             "wire_quotient_full_trajectories": len(quotient),
                             "entropy_trajectories": len(entropy_profiles)})

    feature_matrix = np.asarray([row["features"] for row in quotient_rows])
    scaled = StandardScaler().fit_transform(feature_matrix)
    coordinates = PCA(n_components=2).fit_transform(scaled)
    public_rows = []
    for row, coordinate in zip(quotient_rows, coordinates):
        public_rows.append({"length": row["length"], "multiplicity": row["multiplicity"],
                            "padded": row["padded"], "full_state_pc1": coordinate[0],
                            "full_state_pc2": coordinate[1]})
    public = pd.DataFrame(public_rows)
    summary = pd.DataFrame(summary_rows)
    result = {"inputs": len(permutations), "reachable_semantic_states": len(states),
              "correct_sequence_counts": {str(key): value for key, value in observed.items()},
              "summary": summary_rows, "wire_group_size": math.factorial(n),
              "correct_count_invariant_verified": True}
    return result, public, summary


def main() -> None:
    boolean, prediction, nulls = validate_boolean()
    sorting, trajectories, sorting_summary = validate_sorting()
    artifacts = EXPERIMENT / "artifacts"
    figures = EXPERIMENT / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(artifacts / "boolean_prediction_validation.csv", index=False)
    nulls.to_csv(artifacts / "boolean_null_distributions.csv", index=False)
    trajectories.to_csv(artifacts / "sorting_full_trajectory_atlas.csv", index=False)
    sorting_summary.to_csv(artifacts / "sorting_geometry_summary.csv", index=False)
    result = {"experiment_id": CONFIG["experiment_id"], "boolean": boolean, "sorting": sorting}
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    protocols = ["leave_one_function_out", "leave_one_orbit_out",
                 "leave_one_descriptor_class_out"]
    x = np.arange(3)
    width = 0.24
    for offset, language in enumerate(("NAND", "NOR", "AND_OR_NOT")):
        values = [boolean["languages"][language][protocol]["r2"] for protocol in protocols]
        axes[0].bar(x + (offset - 1) * width, values, width=width,
                    label=language.replace("_", "/"))
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xticks(x, ["function", "input orbit", "descriptor class"])
    axes[0].set_ylabel("held-out $R^2$")
    axes[0].set_title("Prediction under stricter holdouts", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    colors = {"NAND": "#1d6996", "NOR": "#edb120", "AND_OR_NOT": "#d95f02"}
    for language, group in nulls.groupby("language"):
        axes[1].hist(group["null_r2"], bins=35, density=True, alpha=0.38,
                     color=colors[language], label=language.replace("_", "/"))
        observed_r2 = boolean["languages"][language]["leave_one_descriptor_class_out"]["r2"]
        axes[1].axvline(observed_r2, color=colors[language], linewidth=2)
    axes[1].set_xlabel("descriptor-class-held-out $R^2$")
    axes[1].set_ylabel("null density")
    axes[1].set_title("Matched permutation null", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "boolean_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "boolean_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    for column, label, marker in (("entropy_trajectories", "entropy only", "o"),
                                   ("wire_quotient_full_trajectories", "full distribution / wires", "s"),
                                   ("raw_full_state_trajectories", "raw full distribution", "^")):
        axes[0].plot(sorting_summary["length"], sorting_summary[column], marker=marker,
                     label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(5, 9))
    axes[0].set_xlabel("network length")
    axes[0].set_ylabel("distinct semantic trajectories")
    axes[0].set_title("Resolution changes the story", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    scatter = axes[1].scatter(trajectories["full_state_pc1"], trajectories["full_state_pc2"],
                              c=trajectories["length"], cmap="viridis",
                              s=10 + 8 * np.log10(trajectories["multiplicity"]), alpha=0.68,
                              edgecolors="none", rasterized=True)
    axes[1].set_xlabel("full-state PC1")
    axes[1].set_ylabel("full-state PC2")
    axes[1].set_title("Wire-quotiented trajectory atlas", loc="left")
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.05, pad=0.03)
    colorbar.set_label("comparators")
    fig.tight_layout()
    fig.savefig(figures / "sorting_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "sorting_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Results: creativity-atlas validation", "", "## Boolean holdouts and nulls", "",
             "| Language | Function-held-out R² | Orbit-held-out R² | Descriptor-class-held-out R² | Null p |",
             "|---|---:|---:|---:|---:|"]
    for language in ("NAND", "NOR", "AND_OR_NOT"):
        values = boolean["languages"][language]
        lines.append(f"| {language} | {values['leave_one_function_out']['r2']:.3f} | "
                     f"{values['leave_one_orbit_out']['r2']:.3f} | "
                     f"{values['leave_one_descriptor_class_out']['r2']:.3f} | "
                     f"{values['stratified_null']['upper_tail_pvalue']:.4f} |")
    lines.extend(["", "## Sorting geometry", "",
                  "| Length | Entropy profiles | Full trajectories / wire symmetries | Raw full trajectories |",
                  "|---:|---:|---:|---:|"])
    for row in sorting_summary.itertuples():
        lines.append(f"| {row.length} | {row.entropy_trajectories:,} | "
                     f"{row.wire_quotient_full_trajectories:,} | {row.raw_full_state_trajectories:,} |")
    lines.extend(["", "## Interpretation", "",
                  "The strict holdouts test whether semantic prediction extends beyond near-duplicate "
                  "functions; the matched null tests whether it exceeds bias, degree, and symmetry. "
                  "The sorting refinement tests whether the apparent rigidity of optimal entropy "
                  "profiles survives a higher-resolution semantic representation. Conclusions should "
                  "follow these stress tests rather than the original optimistic estimates.", "",
                  "![Boolean validation](figures/boolean_validation.png)", "",
                  "![Sorting validation](figures/sorting_validation.png)"])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
