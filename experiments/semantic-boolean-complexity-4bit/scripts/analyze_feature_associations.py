"""Rank compact semantic invariants by association with exact formula complexity."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"


def load_module(name: str, filename: str):
    path = EXPERIMENT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_cofactor_tables(n: int, variable: int, value: int) -> np.ndarray:
    reduced = np.arange(1 << (n - 1), dtype=np.uint32)
    lower_mask = (1 << variable) - 1
    return ((reduced & lower_mask)
            | ((reduced >> variable) << (variable + 1))
            | (value << variable))


def decision_tree_measures(n: int) -> tuple[np.ndarray, np.ndarray]:
    depths = np.asarray([0, 0], dtype=np.int8)
    leaves = np.asarray([1, 1], dtype=np.int16)
    for variables in range(1, n + 1):
        assignments = 1 << variables
        functions = 1 << assignments
        tables = np.arange(functions, dtype=np.uint32)
        depth_candidates = []
        leaf_candidates = []
        for variable in range(variables):
            cofactors = []
            for value in (0, 1):
                mapped = compact_cofactor_tables(variables, variable, value)
                cofactor = np.sum(
                    ((tables[:, None] >> mapped) & 1)
                    * (1 << np.arange(1 << (variables - 1)))[None, :], axis=1)
                cofactors.append(cofactor.astype(np.int64))
            depth_candidates.append(1 + np.maximum(depths[cofactors[0]], depths[cofactors[1]]))
            leaf_candidates.append(leaves[cofactors[0]] + leaves[cofactors[1]])
        next_depths = np.min(np.stack(depth_candidates), axis=0).astype(np.int8)
        next_leaves = np.min(np.stack(leaf_candidates), axis=0).astype(np.int16)
        next_depths[[0, functions - 1]] = 0
        next_leaves[[0, functions - 1]] = 1
        depths, leaves = next_depths, next_leaves
    return depths, leaves


def factorability_counts(bits: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments = np.arange(1 << n)
    and_count = np.zeros(len(bits), dtype=np.int8)
    or_count = np.zeros(len(bits), dtype=np.int8)
    xor_count = np.zeros(len(bits), dtype=np.int8)
    partitions = [mask for mask in range(1, (1 << n) - 1) if mask & 1]
    for left_mask in partitions:
        left_variables = [i for i in range(n) if left_mask & (1 << i)]
        right_variables = [i for i in range(n) if not left_mask & (1 << i)]
        row_count = 1 << len(left_variables)
        column_count = 1 << len(right_variables)
        mapped = np.empty((row_count, column_count), dtype=int)
        for row in range(row_count):
            for column in range(column_count):
                assignment = 0
                for j, variable in enumerate(left_variables):
                    assignment |= ((row >> j) & 1) << variable
                for j, variable in enumerate(right_variables):
                    assignment |= ((column >> j) & 1) << variable
                mapped[row, column] = assignment
        matrix = bits[:, mapped]
        rows_one = np.any(matrix, axis=2)
        columns_one = np.any(matrix, axis=1)
        and_reconstruction = rows_one[:, :, None] & columns_one[:, None, :]
        nontrivial_and = ((rows_one.sum(axis=1) > 0) & (rows_one.sum(axis=1) < row_count)
                          & (columns_one.sum(axis=1) > 0)
                          & (columns_one.sum(axis=1) < column_count))
        and_count += np.all(matrix == and_reconstruction, axis=(1, 2)) & nontrivial_and

        zeros = ~matrix.astype(bool)
        rows_zero = np.any(zeros, axis=2)
        columns_zero = np.any(zeros, axis=1)
        zero_reconstruction = rows_zero[:, :, None] & columns_zero[:, None, :]
        nontrivial_or = ((rows_zero.sum(axis=1) > 0) & (rows_zero.sum(axis=1) < row_count)
                         & (columns_zero.sum(axis=1) > 0)
                         & (columns_zero.sum(axis=1) < column_count))
        or_count += np.all(zeros == zero_reconstruction, axis=(1, 2)) & nontrivial_or

        row_phase = matrix[:, :, 0] ^ matrix[:, :1, 0]
        column_phase = matrix[:, 0, :] ^ matrix[:, 0, :1]
        xor_reconstruction = row_phase[:, :, None] ^ column_phase[:, None, :] ^ matrix[:, :1, :1]
        nontrivial_xor = ((np.any(row_phase, axis=1)) & (np.any(column_phase, axis=1)))
        xor_count += np.all(matrix == xor_reconstruction, axis=(1, 2)) & nontrivial_xor
    return and_count, or_count, xor_count


def prime_cube_features(bits: np.ndarray, n: int) -> dict[str, np.ndarray]:
    assignments = np.arange(1 << n)
    cubes = list(itertools.product((-1, 0, 1), repeat=n))
    cube_index = {cube: index for index, cube in enumerate(cubes)}
    valid = {output: np.zeros((len(bits), len(cubes)), dtype=bool) for output in (0, 1)}
    literal_counts = np.asarray([sum(value != -1 for value in cube) for cube in cubes])
    for index, cube in enumerate(cubes):
        selected = np.ones(1 << n, dtype=bool)
        for variable, value in enumerate(cube):
            if value != -1:
                selected &= ((assignments >> variable) & 1) == value
        for output in (0, 1):
            valid[output][:, index] = np.all(bits[:, selected] == output, axis=1)
    result = {}
    for output in (0, 1):
        prime = valid[output].copy()
        for index, cube in enumerate(cubes):
            for variable, value in enumerate(cube):
                if value != -1:
                    parent = list(cube)
                    parent[variable] = -1
                    prime[:, index] &= ~valid[output][:, cube_index[tuple(parent)]]
        counts = prime.sum(axis=1)
        literal_sum = prime @ literal_counts
        minimum = np.full(len(bits), n + 1, dtype=np.int8)
        for size in range(n + 1):
            minimum[np.any(prime[:, literal_counts == size], axis=1)] = np.minimum(
                minimum[np.any(prime[:, literal_counts == size], axis=1)], size)
        minimum[counts == 0] = 0
        result[f"prime_{output}_count"] = counts
        result[f"prime_{output}_mean_literals"] = np.divide(
            literal_sum, counts, out=np.zeros(len(bits), dtype=float), where=counts != 0)
        result[f"prime_{output}_minimum_literals"] = minimum
    return result


def induced_component_counts(n: int) -> np.ndarray:
    vertices = 1 << n
    functions = 1 << vertices
    neighbor_masks = [sum(1 << (vertex ^ (1 << variable)) for variable in range(n))
                      for vertex in range(vertices)]
    counts = np.zeros(functions, dtype=np.int8)
    for subset in range(1, functions):
        remaining = subset
        components = 0
        while remaining:
            components += 1
            frontier = remaining & -remaining
            remaining ^= frontier
            while frontier:
                vertex_bit = frontier & -frontier
                frontier ^= vertex_bit
                vertex = vertex_bit.bit_length() - 1
                reached = neighbor_masks[vertex] & remaining
                remaining ^= reached
                frontier |= reached
        counts[subset] = components
    return counts


def canonical_tables(n: int) -> np.ndarray:
    assignments = 1 << n
    functions = 1 << assignments
    tables = np.arange(functions, dtype=np.uint32)
    bits = ((tables[:, None] >> np.arange(assignments)) & 1)
    weights = 1 << np.arange(assignments)
    canonical = np.full(functions, functions - 1, dtype=np.uint32)
    for permutation in itertools.permutations(range(n)):
        mapped = np.asarray([
            sum(((assignment >> permutation[index]) & 1) << index for index in range(n))
            for assignment in range(assignments)
        ])
        transformed = np.sum(bits[:, mapped] * weights[None, :], axis=1).astype(np.uint32)
        canonical = np.minimum(canonical, transformed)
    return canonical


def correlation_ratio(target: np.ndarray, feature: np.ndarray) -> tuple[float, float, int]:
    _, groups = np.unique(feature, return_inverse=True)
    levels = int(groups.max() + 1)
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=target) / counts
    prediction = means[groups]
    total = np.sum((target - target.mean()) ** 2)
    eta2 = float(np.sum((prediction - target.mean()) ** 2) / total)
    adjusted = float(1 - (1 - eta2) * (len(target) - 1) / max(1, len(target) - levels))
    return eta2, adjusted, levels


def class_mean_r2(target: np.ndarray, values: np.ndarray) -> float:
    _, groups = np.unique(values, axis=0, return_inverse=True)
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=target) / counts
    prediction = means[groups]
    return float(1 - np.sum((target - prediction) ** 2)
                 / np.sum((target - target.mean()) ** 2))


def main() -> None:
    run = load_module("semantic_boolean_run", "run.py")
    gap = load_module("semantic_boolean_gap", "analyze_gap.py")
    semantic, raw, _ = run.truth_table_features()
    candidates = gap.candidate_invariants(run.N)
    tables = semantic.truth_table.to_numpy(dtype=np.uint32)
    assignments = np.arange(run.ASSIGNMENTS, dtype=np.uint32)
    bits = ((tables[:, None] >> assignments) & 1).astype(np.int8)
    signs = 2 * bits.astype(np.int16) - 1
    degrees = np.asarray([int(mask).bit_count() for mask in assignments])
    walsh = gap.transform_rows(signs)
    features = semantic[["truth_table", *run.FEATURES]].copy()
    metadata = []

    def register(name: str, values: np.ndarray, status: str, family: str,
                 rationale: str, evidence: str) -> None:
        features[name] = np.asarray(values)
        metadata.append({"feature": name, "status": status, "family": family,
                         "rationale": rationale, "evidence": evidence})

    original_rationales = {
        "bias": ("output balance", "Extreme bias gives short canonical DNF/CNF constructions and conditions sensitivity bounds.", "indirect bound"),
        "influence": ("influence", "Khrapchenko-type bounds turn high sensitivity into De Morgan formula lower bounds.", "direct lower-bound precedent"),
        "fourier": ("Fourier magnitude", "Small De Morgan formulas have low-degree Fourier concentration.", "direct lower-bound precedent"),
        "algebraic_degree": ("ANF", "Binary gates can increase GF(2) degree only compositionally, giving a weak gate lower bound.", "elementary lower bound"),
        "input_symmetries": ("symmetry", "Invariance groups constrain circuit classes, although stabilizer cardinality alone is coarse.", "indirect precedent"),
    }
    for name in run.FEATURES:
        key = ("influence" if name.startswith("influence_") else
               "fourier" if name.startswith("fourier_energy_") else name)
        family, rationale, evidence = original_rationales[key]
        metadata.append({"feature": name, "status": "used", "family": family,
                         "rationale": rationale, "evidence": evidence})

    for degree in range(run.N + 1):
        register(f"anf_terms_degree_{degree}", candidates["anf_degree_profile"][:, degree],
                 "new", "ANF support", "Counts algebraic support discarded by degree alone; gate composition acts explicitly on ANF polynomials.", "compositional hypothesis")
        register(f"fourier_l1_degree_{degree}", candidates["fourier_l1_by_degree"][:, degree],
                 "new", "Fourier shape", "L1 mass refines degree-wise L2 energy and measures spectral concentration within a degree.", "Fourier precedent")
        register(f"fourier_negative_degree_{degree}", candidates["fourier_sign_profile"][:, 3 * degree],
                 "new", "Fourier phase", "Counts coefficient signs discarded by Fourier energy; controlled collisions show basis-specific phase sensitivity.", "experimental/compositional")
        register(f"fourier_zero_degree_{degree}", candidates["fourier_sign_profile"][:, 3 * degree + 1],
                 "new", "Fourier shape", "Measures spectral sparsity within a degree, which energy totals cannot recover.", "Fourier heuristic")

    local_sensitivity = np.zeros_like(bits)
    directional = {order: [] for order in (2, 3)}
    for flip in range(1, 1 << run.N):
        changed = np.mean(bits != bits[:, assignments ^ flip], axis=1)
        if flip.bit_count() == 1:
            variable = int(math.log2(flip))
            local_sensitivity += bits != bits[:, assignments ^ (1 << variable)]
        elif flip.bit_count() in directional:
            directional[flip.bit_count()].append(changed)
    counts_zero = np.sum(bits == 0, axis=1)
    counts_one = np.sum(bits == 1, axis=1)
    sensitivity_zero = np.divide(np.sum(local_sensitivity * (bits == 0), axis=1), counts_zero,
                                 out=np.zeros(len(bits)), where=counts_zero != 0)
    sensitivity_one = np.divide(np.sum(local_sensitivity * (bits == 1), axis=1), counts_one,
                                out=np.zeros(len(bits)), where=counts_one != 0)
    register("sensitivity_variance", np.var(local_sensitivity, axis=1), "new", "local sensitivity",
             "Retains heterogeneity across inputs beyond the mean influence used by classical bounds.", "refinement of direct precedent")
    register("maximum_local_sensitivity", np.max(local_sensitivity, axis=1), "new", "local sensitivity",
             "Worst-case sensitivity is a standard Boolean complexity measure related to other query measures.", "indirect precedent")
    register("zero_conditional_sensitivity", sensitivity_zero, "new", "local sensitivity",
             "Average boundary degree on zero-inputs is one factor in Khrapchenko's formula lower bound.", "direct lower-bound term")
    register("one_conditional_sensitivity", sensitivity_one, "new", "local sensitivity",
             "Average boundary degree on one-inputs is one factor in Khrapchenko's formula lower bound.", "direct lower-bound term")
    register("khrapchenko_product", sensitivity_zero * sensitivity_one, "new", "local sensitivity",
             "This product directly lower-bounds De Morgan formula leaf size.", "direct lower bound")
    for order, values in directional.items():
        matrix = np.stack(values, axis=1)
        register(f"order_{order}_derivative_std", np.round(np.std(matrix, axis=1), 12), "new", "higher derivatives",
                 "Measures anisotropy of multi-bit derivatives beyond ordinary single-variable influence.", "Boolean-analysis heuristic")
        register(f"order_{order}_derivative_max", np.max(matrix, axis=1), "new", "higher derivatives",
                 "Captures the strongest multi-bit perturbation direction, which degree-wise energy averages away.", "Boolean-analysis heuristic")

    certificate = candidates["certificate_profile"].reshape(len(bits), 2, run.N + 1)
    sizes = np.arange(run.N + 1)
    certificate_total = certificate.sum(axis=1)
    certificate_mean = (certificate_total @ sizes) / run.ASSIGNMENTS
    certificate_max = np.max(np.where(certificate_total > 0, sizes[None, :], 0), axis=1)
    register("mean_certificate_size", certificate_mean, "new", "certificates",
             "Certificates measure how many inputs suffice to force the output and bound decision-tree complexity.", "indirect complexity precedent")
    register("maximum_certificate_size", certificate_max, "new", "certificates",
             "Worst-case certificate complexity is polynomially related to standard query-complexity measures.", "indirect complexity precedent")
    for output in (0, 1):
        denominator = certificate[:, output, :].sum(axis=1)
        mean = np.divide(certificate[:, output, :] @ sizes, denominator,
                         out=np.zeros(len(bits)), where=denominator != 0)
        register(f"mean_certificate_size_output_{output}", mean, "new", "certificates",
                 "Output-conditioned certificates distinguish the two sides of the truth-table boundary.", "indirect complexity precedent")

    for codimension in range(1, run.N):
        profile = candidates["constant_restriction_profile"][:, 2 * (codimension - 1):2 * codimension]
        register(f"constant_faces_codimension_{codimension}", profile.sum(axis=1), "new", "restrictions",
                 "Counts restrictions that collapse the function; formula shrinkage under restrictions is a classical lower-bound method.", "direct methodological precedent")

    depth, leaves = decision_tree_measures(run.N)
    register("decision_tree_depth", depth, "new", "decision trees",
             "Exact decision-tree depth is linked by known inequalities to sensitivity, degree, and certificates.", "indirect complexity precedent")
    register("decision_tree_leaf_count", leaves, "new", "decision trees",
             "Decision-tree leaves measure a tree representation cost and give constructive Boolean formulas.", "constructive upper-bound precedent")

    and_factors, or_factors, xor_factors = factorability_counts(bits, run.N)
    register("and_factorable_partitions", and_factors, "new", "factorability",
             "A nontrivial AND decomposition realizes the formula recurrence as two smaller independent subformulas.", "direct compositional rationale")
    register("or_factorable_partitions", or_factors, "new", "factorability",
             "A nontrivial OR decomposition realizes the formula recurrence as two smaller independent subformulas.", "direct compositional rationale")
    register("xor_factorable_partitions", xor_factors, "new", "factorability",
             "XOR decomposition measures parity structure that must be simulated in gate bases lacking XOR.", "direct compositional rationale")

    for name, values in prime_cube_features(bits, run.N).items():
        register(name, values, "new", "prime cubes",
                 "Prime implicants/implicates determine canonical DNF/CNF constructions and hence explicit formula upper bounds.", "constructive upper-bound precedent")

    components = induced_component_counts(run.N)
    one_components = components[tables]
    zero_components = components[(run.FUNCTIONS - 1) ^ tables]
    register("one_region_components", one_components, "new", "cube topology",
             "Counts disconnected one-regions; it refines the hypercube boundary geometry measured by influence.", "geometric heuristic")
    register("zero_region_components", zero_components, "new", "cube topology",
             "Counts disconnected zero-regions; it refines the hypercube boundary geometry measured by influence.", "geometric heuristic")
    register("total_monochromatic_components", one_components + zero_components, "new", "cube topology",
             "Measures fragmentation of both output regions beyond their shared boundary size.", "geometric heuristic")

    power = walsh.astype(float) ** 2
    probabilities = power / np.sum(power, axis=1, keepdims=True)
    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    register("maximum_absolute_walsh", np.max(np.abs(walsh), axis=1), "new", "Fourier shape",
             "The largest Walsh coefficient measures proximity to an affine character and refines energy totals.", "Fourier precedent")
    register("fourier_entropy", entropy, "new", "Fourier shape",
             "Spectral entropy measures how diffusely Fourier mass is distributed rather than only where by degree.", "Fourier heuristic")

    canonical_three = canonical_tables(run.N - 1)
    cofactor_orbits = []
    canalizing = np.zeros(len(bits), dtype=np.int8)
    for variable in range(run.N):
        variable_cofactors = []
        for value in (0, 1):
            mapped = compact_cofactor_tables(run.N, variable, value)
            compact = np.sum(bits[:, mapped] * (1 << np.arange(1 << (run.N - 1)))[None, :], axis=1)
            variable_cofactors.append(compact.astype(np.int64))
            cofactor_orbits.append(canonical_three[compact.astype(np.int64)])
        constant = ((variable_cofactors[0] == 0) | (variable_cofactors[0] == 255)
                    | (variable_cofactors[1] == 0) | (variable_cofactors[1] == 255))
        canalizing += constant
    sorted_cofactors = np.sort(np.stack(cofactor_orbits, axis=1), axis=1)
    cofactor_diversity = 1 + np.sum(sorted_cofactors[:, 1:] != sorted_cofactors[:, :-1], axis=1)
    register("canalizing_variable_count", canalizing, "new", "restrictions",
             "Canalizing variables yield constant cofactors and immediate Shannon/formula simplifications.", "constructive rationale")
    register("cofactor_orbit_diversity", cofactor_diversity, "new", "restrictions",
             "Counts distinct restricted subproblems modulo input renaming; fewer types suggest reusable recursive structure.", "restriction heuristic")

    predictions = pd.read_csv(ARTIFACTS / "predictions.csv")
    languages = run.CONFIG["languages"]
    targets = {language: predictions[f"{language}_minimum_gates"].to_numpy(float)
               for language in languages}
    for language, target in targets.items():
        features[f"target_{language}"] = target.astype(int)
    features.to_csv(ARTIFACTS / "feature_table.csv", index=False)
    metadata_frame = pd.DataFrame(metadata)
    source_by_family = {
        "output balance": "[7] empirical bias/complexity precedent",
        "influence": "[1] Khrapchenko-type formula lower bounds",
        "Fourier magnitude": "[2] Fourier concentration for formulas",
        "ANF": "elementary gate-composition bound",
        "symmetry": "[5] invariance-group complexity theory",
        "ANF support": "elementary GF(2) gate composition; present hypothesis",
        "Fourier shape": "[2] Fourier concentration; refinement proposed here",
        "Fourier phase": "present controlled-intervention evidence",
        "local sensitivity": "[1] Khrapchenko-type formula lower bounds",
        "higher derivatives": "[1,3] Boolean/query complexity precedent",
        "certificates": "[3] decision-tree complexity survey",
        "restrictions": "[4] formula shrinkage under restrictions",
        "decision trees": "[3] decision-tree complexity survey",
        "factorability": "minimum-formula recurrence; present hypothesis",
        "prime cubes": "[6] prime implicants and canonical DNF/CNF",
        "cube topology": "[1] boundary/sensitivity identity; present refinement",
    }
    metadata_frame["source"] = metadata_frame.family.map(source_by_family)
    metadata_frame.to_csv(ARTIFACTS / "feature_definitions.csv", index=False)

    base_ceiling = {language: class_mean_r2(target, raw) for language, target in targets.items()}
    association_rows = []
    feature_names = metadata_frame.feature.tolist()
    for name in feature_names:
        values = features[name].to_numpy()
        for language, target in targets.items():
            eta2, adjusted_eta2, levels = correlation_ratio(target, values)
            augmented = np.column_stack((raw, values))
            conditional_gain = class_mean_r2(target, augmented) - base_ceiling[language]
            association_rows.append({
                "feature": name,
                "status": metadata_frame.set_index("feature").loc[name, "status"],
                "family": metadata_frame.set_index("feature").loc[name, "family"],
                "language": language,
                "distinct_values": levels,
                "pearson_r": float(pd.Series(values).corr(pd.Series(target), method="pearson")),
                "spearman_rho": float(pd.Series(values).corr(pd.Series(target), method="spearman")),
                "eta_squared": eta2,
                "adjusted_eta_squared": adjusted_eta2,
                "expected_null_eta_squared": float((levels - 1) / (len(target) - 1)),
                "conditional_ceiling_gain": conditional_gain,
            })
    associations = pd.DataFrame(association_rows)
    associations.to_csv(ARTIFACTS / "feature_target_associations.csv", index=False)

    aggregate = (associations.groupby(["feature", "status", "family"], as_index=False)
                 .agg(mean_adjusted_eta_squared=("adjusted_eta_squared", "mean"),
                      mean_absolute_spearman=("spearman_rho", lambda values: np.mean(np.abs(values))),
                      mean_conditional_ceiling_gain=("conditional_ceiling_gain", "mean"),
                      distinct_values=("distinct_values", "max")))
    aggregate = aggregate.sort_values("mean_adjusted_eta_squared", ascending=False)
    aggregate.to_csv(ARTIFACTS / "feature_target_ranking.csv", index=False)

    # Strict validation for the best compact new features by conditional information.
    compact = aggregate[(aggregate.status == "new") & (aggregate.distinct_values <= 20)]
    validation_names = compact.nlargest(6, "mean_conditional_ceiling_gain").feature.tolist()
    validation_rows = []
    for name in ["current_descriptor", *validation_names]:
        values = raw if name == "current_descriptor" else np.column_stack((raw, features[name]))
        _, groups = np.unique(values, axis=0, return_inverse=True)
        scaled = StandardScaler().fit_transform(values)
        tests, neighborhoods = run.tied_neighbor_plan(scaled, groups, groups, run.K)
        for language, target in targets.items():
            metrics = run.evaluate_tied(target, tests, neighborhoods)
            validation_rows.append({"feature_addition": name, "language": language,
                                    "descriptor_classes": int(groups.max() + 1),
                                    "strict_r2": metrics["r2"], "strict_rmse": metrics["rmse"]})
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(ARTIFACTS / "feature_strict_validation.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    top_count = 25
    plot_features = aggregate.head(top_count).iloc[::-1]
    conditional_features = aggregate.sort_values("mean_conditional_ceiling_gain", ascending=False).head(top_count).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    language_colors = {"NAND": "#40566A", "NOR": "#87939C", "AND_OR_NOT": "#25292C"}
    for axis, frame, metric, title in (
        (axes[0], plot_features, "adjusted_eta_squared", "Standalone nonlinear association"),
        (axes[1], conditional_features, "conditional_ceiling_gain", "New information beyond original descriptor"),
    ):
        names = frame.feature.tolist()
        means = (associations[associations.feature.isin(names)]
                 .groupby("feature")[metric].mean().reindex(names))
        colors = ["#B98249" if frame.set_index("feature").loc[name, "status"] == "new" else "#7A8C99"
                  for name in names]
        axis.barh(np.arange(len(names)), means, color=colors, alpha=0.55)
        offsets = {"NAND": -0.18, "NOR": 0.0, "AND_OR_NOT": 0.18}
        for language in languages:
            subset = associations[(associations.language == language)].set_index("feature")
            axis.scatter(subset.loc[names, metric], np.arange(len(names)) + offsets[language],
                         s=20, color=language_colors[language], label=language)
        labels = [f"{'NEW' if frame.set_index('feature').loc[name, 'status'] == 'new' else 'USED'}  {name}"
                  for name in names]
        axis.set_yticks(np.arange(len(names)), labels, fontsize=8)
        axis.set_xlabel("adjusted eta squared" if metric == "adjusted_eta_squared" else "descriptor-ceiling R2 gain")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#D9DDDF", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Semantic features associated with exact formula complexity", x=0.06,
                 ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"feature_target_associations.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ranking_table = aggregate.head(30).copy()
    ranking_table.columns = [column.replace("_", " ") for column in ranking_table.columns]
    validation_table = validation.pivot(index="feature_addition", columns="language", values="strict_r2").reset_index()
    definitions = metadata_frame.merge(aggregate[["feature", "mean_adjusted_eta_squared",
                                                   "mean_conditional_ceiling_gain"]], on="feature")
    definitions = definitions.sort_values(["status", "family", "feature"])
    sources = [
        "[1] [Khrapchenko/sensitivity presentation in Jukna's *Boolean Function Complexity*](https://web.vu.lt/mif/s.jukna/boolean/bool-V7.pdf)",
        "[2] [Impagliazzo and Kabanets, *Fourier Concentration from Shrinkage*](https://www2.cs.sfu.ca/~kabanets/papers/Fourier-final-CC.pdf)",
        "[3] [Buhrman and de Wolf, *Complexity Measures and Decision Tree Complexity*](https://homepages.cwi.nl/~rdewolf/publ/qc/dectree.pdf)",
        "[4] [Impagliazzo, *The Effect of Random Restrictions on Formula Size*](https://doi.org/10.1002/rsa.3240040202)",
        "[5] [Babai, Beals, and Takacsi-Nagy, *Symmetry and Complexity*](https://doi.org/10.1145/129712.129754)",
        "[6] [Chandra and Markowsky, *On the Number of Prime Implicants*](https://doi.org/10.1016/0012-365X(78)90168-1)",
        "[7] [Gherardi and Rotondo, *Measuring Logic Complexity Can Guide Pattern Discovery*](https://arxiv.org/abs/1603.03337)",
    ]
    lines = [
        "# Feature association with exact Boolean formula complexity", "",
        "The ranking covers all 65,536 four-input functions. The primary statistic is",
        "adjusted correlation ratio (eta squared): the fraction of gate-count variance",
        "explained by a scalar feature without assuming a linear or monotone relation,",
        "adjusted for the feature's number of distinct values. Spearman rho records",
        "monotone association. Conditional ceiling gain asks how much information the",
        "feature adds beyond the paper's complete original descriptor. These are",
        "population descriptions, not causal effects.", "",
        "![Sorted feature associations](figures/feature_target_associations.png)", "",
        "## Overall ranking", "",
        ranking_table.to_markdown(index=False, floatfmt=".4f"), "",
        "## Strict validation of leading compact additions", "",
        validation_table.to_markdown(index=False, floatfmt=".4f"), "",
        "## Feature definitions and mathematical rationale", "",
        "`used` means the feature was in the paper; `new` means it was added here.",
        "Evidence labels distinguish direct bounds from indirect precedent and",
        "exploratory hypotheses.", "",
        definitions[["feature", "status", "family", "evidence", "source", "rationale",
                     "mean_adjusted_eta_squared", "mean_conditional_ceiling_gain"]].to_markdown(
                         index=False, floatfmt=".4f"), "",
        "## Sources", "", *sources, "",
        "## Reproduce", "", "```powershell",
        "python experiments/semantic-boolean-complexity-4bit/scripts/analyze_feature_associations.py",
        "```", "",
    ]
    (EXPERIMENT / "FEATURE_ASSOCIATIONS.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"features": len(feature_names), "used": int((metadata_frame.status == "used").sum()),
              "new": int((metadata_frame.status == "new").sum()),
              "strictly_validated_additions": validation_names,
              "top_overall": aggregate.head(10).feature.tolist()}
    (ARTIFACTS / "feature_association_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
