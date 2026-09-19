"""Exact four-input synthesis and syntax-free complexity prediction."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler


EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
N = CONFIG["variables"]
ASSIGNMENTS = 1 << N
FUNCTIONS = 1 << ASSIGNMENTS
MASK = FUNCTIONS - 1
K = CONFIG["neighbors"]
NAVY = "#40566A"
SLATE = "#87939C"
LIGHT_GRAY = "#D9DDDF"
CHARCOAL = "#25292C"
LANGUAGE_COLORS = [NAVY, SLATE, CHARCOAL]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": CHARCOAL, "axes.labelcolor": CHARCOAL,
                     "text.color": CHARCOAL, "xtick.color": CHARCOAL,
                     "ytick.color": CHARCOAL, "savefig.facecolor": "white"})


def feature_names(n: int) -> list[str]:
    return (["bias"] + [f"influence_{i}" for i in range(n)]
            + [f"fourier_energy_degree_{i}" for i in range(n + 1)]
            + ["algebraic_degree", "input_symmetries"])


FEATURES = feature_names(N)


def variable_tables(n: int) -> np.ndarray:
    return np.asarray([
        sum(((assignment >> variable) & 1) << assignment
            for assignment in range(1 << n))
        for variable in range(n)
    ], dtype=np.uint32)


def synthesize_minima(language: str, n: int = N) -> tuple[np.ndarray, list[int]]:
    """Enumerate exact minimum gate counts, retaining only minimum subformulas."""
    assignments = 1 << n
    functions = 1 << assignments
    mask = functions - 1
    variables = variable_tables(n)
    minimum = np.full(functions, -1, dtype=np.int16)
    minimum[variables] = 0
    layers = [variables]
    layer_sizes = [len(variables)]

    for cost in range(1, 128):
        candidate = np.zeros(functions, dtype=bool)
        if language == "AND_OR_NOT":
            candidate[(~layers[cost - 1]) & mask] = True

        # Every binary gate is commutative, so one ordering of cost partitions suffices.
        for left_cost in range((cost - 1) // 2 + 1):
            right_cost = cost - 1 - left_cost
            if right_cost >= len(layers):
                continue
            left = layers[left_cost]
            right = layers[right_cost]
            for start in range(0, len(left), 512):
                block = left[start:start + 512, None]
                if language == "NAND":
                    outputs = ((~(block & right[None, :])) & mask,)
                elif language == "NOR":
                    outputs = ((~(block | right[None, :])) & mask,)
                elif language == "AND_OR_NOT":
                    outputs = (block & right[None, :], block | right[None, :])
                else:
                    raise ValueError(language)
                for output in outputs:
                    candidate[np.unique(output)] = True

        candidate[minimum >= 0] = False
        new = np.flatnonzero(candidate).astype(np.uint32)
        minimum[new] = cost
        layers.append(new)
        layer_sizes.append(len(new))
        if np.all(minimum >= 0):
            return minimum, layer_sizes
    raise RuntimeError(f"{language} exceeded the synthesis bound")


def truth_table_features(n: int = N) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute exact integer encodings of the invariant semantic descriptor."""
    assignments = 1 << n
    functions = 1 << assignments
    tables = np.arange(functions, dtype=np.uint32)
    assignment_ids = np.arange(assignments, dtype=np.uint32)
    bits = ((tables[:, None] >> assignment_ids) & 1).astype(np.int16)
    signs = 2 * bits - 1

    columns: dict[str, np.ndarray] = {"bias": signs.sum(axis=1)}
    influences = []
    for variable in range(n):
        changed = np.sum(bits != bits[:, assignment_ids ^ (1 << variable)], axis=1)
        influences.append(changed)
    ordered = np.sort(np.stack(influences, axis=1), axis=1)
    for index in range(n):
        columns[f"influence_{index}"] = ordered[:, index]

    walsh = signs.copy()
    stride = 1
    while stride < assignments:
        for start in range(0, assignments, 2 * stride):
            left = walsh[:, start:start + stride].copy()
            right = walsh[:, start + stride:start + 2 * stride].copy()
            walsh[:, start:start + stride] = left + right
            walsh[:, start + stride:start + 2 * stride] = left - right
        stride *= 2
    degrees = np.asarray([mask.bit_count() for mask in range(assignments)])
    for degree in range(n + 1):
        columns[f"fourier_energy_degree_{degree}"] = np.sum(
            walsh[:, degrees == degree] ** 2, axis=1)

    anf = bits.copy()
    for variable in range(n):
        for monomial in range(assignments):
            if monomial & (1 << variable):
                anf[:, monomial] ^= anf[:, monomial ^ (1 << variable)]
    columns["algebraic_degree"] = np.max(anf * degrees[None, :], axis=1)

    symmetries = np.zeros(functions, dtype=np.int16)
    orbit_canonical = np.full(functions, functions - 1, dtype=np.uint32)
    weights = 1 << assignment_ids
    for permutation in itertools.permutations(range(n)):
        mapped = np.asarray([
            sum(((assignment >> permutation[index]) & 1) << index
                for index in range(n))
            for assignment in range(assignments)
        ])
        symmetries += np.all(bits == bits[:, mapped], axis=1)
        transformed = np.sum(bits[:, mapped] * weights[None, :], axis=1).astype(np.uint32)
        orbit_canonical = np.minimum(orbit_canonical, transformed)
    columns["input_symmetries"] = symmetries

    frame = pd.DataFrame({"truth_table": tables, **columns})
    raw = frame[feature_names(n)].to_numpy(dtype=float)
    return frame, raw, orbit_canonical


def descriptor_classes(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(raw, axis=0, return_inverse=True)
    return unique, inverse


def tied_neighbor_plan(features: np.ndarray, class_ids: np.ndarray,
                       groups: np.ndarray, neighbors: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Select at least k neighbors and include the complete kth-distance shell."""
    representatives = np.asarray([
        features[np.flatnonzero(class_ids == class_id)[0]]
        for class_id in range(class_ids.max() + 1)
    ])
    members = [np.flatnonzero(class_ids == class_id)
               for class_id in range(class_ids.max() + 1)]
    tests: list[np.ndarray] = []
    neighborhoods: list[np.ndarray] = []
    class_index = np.arange(len(representatives))
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        source_class = int(class_ids[test[0]])
        distances = np.sum((representatives - representatives[source_class]) ** 2, axis=1)
        order = np.lexsort((class_index, distances))
        selected: list[np.ndarray] = []
        selected_count = 0
        position = 0
        while position < len(order) and selected_count < neighbors:
            distance = distances[order[position]]
            shell_end = position + 1
            while (shell_end < len(order)
                   and np.isclose(distances[order[shell_end]], distance,
                                  rtol=1e-12, atol=1e-12)):
                shell_end += 1
            for candidate_class in order[position:shell_end]:
                eligible = members[int(candidate_class)]
                eligible = eligible[groups[eligible] != group]
                if len(eligible):
                    selected.append(eligible)
                    selected_count += len(eligible)
            position = shell_end
        if selected_count < neighbors:
            raise RuntimeError(f"only {selected_count} tied neighbors for group {group}")
        tests.append(test)
        neighborhoods.append(np.concatenate(selected))
    return tests, neighborhoods


def evaluate_tied(target: np.ndarray, tests: list[np.ndarray],
                  neighborhoods: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    prediction = np.empty_like(target, dtype=float)
    for test, neighborhood in zip(tests, neighborhoods):
        prediction[test] = target[neighborhood].mean()
    return {
        "r2": float(r2_score(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "prediction": prediction,
    }


def stratified_null_tied(target: np.ndarray, tests: list[np.ndarray],
                         neighborhoods: list[np.ndarray], strata: np.ndarray,
                         repetitions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stratum_members = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    values = np.empty(repetitions, dtype=float)
    prediction = np.empty_like(target, dtype=float)
    for repetition in range(repetitions):
        shuffled = target.copy()
        for indices in stratum_members:
            shuffled[indices] = rng.permutation(shuffled[indices])
        for test, neighborhood in zip(tests, neighborhoods):
            prediction[test] = shuffled[neighborhood].mean()
        residual = np.sum((shuffled - prediction) ** 2)
        total = np.sum((shuffled - shuffled.mean()) ** 2)
        values[repetition] = 1 - residual / total
    return values


def verify_three_bit() -> None:
    reference_path = (ROOT / "experiments" / "creativity-boolean-atlas" /
                      "artifacts" / "boolean_atlas.csv")
    reference = pd.read_csv(reference_path)
    for language in CONFIG["languages"]:
        expected = (reference[reference["language"] == language]
                    .sort_values("truth_table")["minimum_gates"].to_numpy())
        observed, _ = synthesize_minima(language, n=3)
        if not np.array_equal(expected, observed):
            raise RuntimeError(f"three-bit regression failed for {language}")


def main() -> None:
    started = time.time()
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    semantic, raw, orbit_canonical = truth_table_features()
    unique_descriptors, class_ids = descriptor_classes(raw)
    scaled = StandardScaler().fit_transform(raw)
    function_groups = np.arange(FUNCTIONS)
    _, orbit_groups = np.unique(orbit_canonical, return_inverse=True)

    protocol_groups = {
        "function_holdout": function_groups,
        "orbit_holdout": orbit_groups,
        "descriptor_holdout": class_ids,
    }
    tied_plans = {name: tied_neighbor_plan(scaled, class_ids, groups, K)
                  for name, groups in protocol_groups.items()}
    strict_tests, strict_neighborhoods = tied_plans["descriptor_holdout"]

    feature_sets = {
        "full": FEATURES,
        "bias_degree_symmetry": ["bias", "algebraic_degree", "input_symmetries"],
        "influences_only": [f"influence_{i}" for i in range(N)],
        "fourier_only": [f"fourier_energy_degree_{i}" for i in range(N + 1)],
        "without_influences": [name for name in FEATURES if not name.startswith("influence_")],
        "without_fourier": [name for name in FEATURES
                            if not name.startswith("fourier_energy_")],
    }
    ablation_plans = {}
    for name, columns in feature_sets.items():
        values = StandardScaler().fit_transform(semantic[columns].to_numpy(dtype=float))
        ablation_plans[name] = tied_neighbor_plan(values, class_ids, class_ids, K)

    z = semantic[["bias", "algebraic_degree", "input_symmetries"]].to_numpy()
    _, strata = np.unique(z, axis=0, return_inverse=True)

    minima = {}
    layers = {}
    for language in CONFIG["languages"]:
        minima[language], layer_sizes = synthesize_minima(language)
        layers[language] = layer_sizes
    verify_three_bit()

    # NAND/NOR duality: K_NAND(f) = K_NOR(not f(not x)).
    input_complement = np.arange(FUNCTIONS, dtype=np.uint32)
    bits = ((input_complement[:, None] >> np.arange(ASSIGNMENTS)) & 1)
    mapped_assignments = np.arange(ASSIGNMENTS) ^ (ASSIGNMENTS - 1)
    dual = MASK ^ np.sum(bits[:, mapped_assignments]
                         * (1 << np.arange(ASSIGNMENTS))[None, :], axis=1).astype(np.uint32)
    if not np.array_equal(minima["NAND"], minima["NOR"][dual]):
        raise RuntimeError("NAND/NOR duality invariant failed")

    prediction_rows = []
    ablation_rows = []
    null_rows = []
    results = {
        "experiment_id": CONFIG["experiment_id"],
        "functions": FUNCTIONS,
        "input_permutation_orbits": int(len(np.unique(orbit_groups))),
        "descriptor_classes": int(len(unique_descriptors)),
        "neighbors": K,
        "tie_policy": "include the complete kth-distance shell",
        "languages": {},
        "checks": {"three_bit_regression": True, "nand_nor_duality": True},
    }

    predictions_frame = pd.DataFrame({"truth_table": np.arange(FUNCTIONS)})
    for language_index, language in enumerate(CONFIG["languages"]):
        target = minima[language].astype(float)
        predictions_frame[f"{language}_minimum_gates"] = target.astype(int)
        language_result = {
            "maximum_minimum_gates": int(target.max()),
            "mean_minimum_gates": float(target.mean()),
            "protocols": {},
            "ablations_descriptor_holdout": {},
        }
        descriptor_oracle = np.empty_like(target)
        for descriptor_class in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == descriptor_class)
            descriptor_oracle[indices] = target[indices].mean()
        language_result["descriptor_class_mean_ceiling_r2"] = float(
            r2_score(target, descriptor_oracle))
        for protocol, (tests, neighborhoods) in tied_plans.items():
            metrics = evaluate_tied(target, tests, neighborhoods)
            language_result["protocols"][protocol] = {
                "r2": metrics["r2"], "rmse": metrics["rmse"]}
            predictions_frame[f"{language}_{protocol}_prediction"] = metrics["prediction"]
            prediction_rows.append({"language": language, "protocol": protocol,
                                    "r2": metrics["r2"], "rmse": metrics["rmse"]})

        for ablation, (tests, neighborhoods) in ablation_plans.items():
            metrics = evaluate_tied(target, tests, neighborhoods)
            language_result["ablations_descriptor_holdout"][ablation] = {
                "r2": metrics["r2"], "rmse": metrics["rmse"]}
            ablation_rows.append({"language": language, "features": ablation,
                                  "r2": metrics["r2"], "rmse": metrics["rmse"]})

        strict_observed = language_result["protocols"]["descriptor_holdout"]["r2"]
        null = stratified_null_tied(
            target, strict_tests, strict_neighborhoods, strata,
            CONFIG["null_repetitions"], CONFIG["seed"] + language_index)
        language_result["matched_null"] = {
            "repetitions": CONFIG["null_repetitions"],
            "mean_r2": float(null.mean()),
            "q025_r2": float(np.quantile(null, 0.025)),
            "q975_r2": float(np.quantile(null, 0.975)),
            "upper_tail_pvalue": float((1 + np.sum(null >= strict_observed)) /
                                        (len(null) + 1)),
        }
        null_rows.extend({"language": language, "repetition": index,
                          "null_r2": value} for index, value in enumerate(null))
        language_result["layer_sizes"] = layers[language]
        results["languages"][language] = language_result

    matrix = np.column_stack([minima[language] for language in CONFIG["languages"]])
    results["cross_language_spearman"] = {}
    for left, right in itertools.combinations(range(len(CONFIG["languages"])), 2):
        rho = float(pd.Series(matrix[:, left]).corr(pd.Series(matrix[:, right]), method="spearman"))
        key = f"{CONFIG['languages'][left]}__{CONFIG['languages'][right]}"
        results["cross_language_spearman"][key] = rho

    # Apples-to-apples three-bit reanalysis with the same tie-aware rule.
    three_semantic, three_raw, three_orbit_canonical = truth_table_features(n=3)
    _, three_classes = descriptor_classes(three_raw)
    three_scaled = StandardScaler().fit_transform(three_raw)
    _, three_orbits = np.unique(three_orbit_canonical, return_inverse=True)
    three_groups = {
        "function_holdout": np.arange(256),
        "orbit_holdout": three_orbits,
        "descriptor_holdout": three_classes,
    }
    three_plans = {name: tied_neighbor_plan(three_scaled, three_classes, groups, K)
                   for name, groups in three_groups.items()}
    three_reference = pd.read_csv(
        ROOT / "experiments" / "creativity-boolean-atlas" / "artifacts" /
        "boolean_atlas.csv")
    results["three_bit_tie_aware"] = {}
    three_rows = []
    for language in CONFIG["languages"]:
        target = (three_reference[three_reference["language"] == language]
                  .sort_values("truth_table")["minimum_gates"].to_numpy(dtype=float))
        results["three_bit_tie_aware"][language] = {}
        for protocol, (tests, neighborhoods) in three_plans.items():
            metrics = evaluate_tied(target, tests, neighborhoods)
            compact = {"r2": metrics["r2"], "rmse": metrics["rmse"]}
            results["three_bit_tie_aware"][language][protocol] = compact
            three_rows.append({"bits": 3, "language": language,
                               "protocol": protocol, **compact})

    # Sensitivity of the strict four-bit result to k and coordinate scaling.
    robustness_rows = []
    results["robustness"] = {"k_sweep": {}, "scaling": {}}
    for neighbors in (3, 5, 7, 11, 15):
        tests, neighborhoods = tied_neighbor_plan(
            scaled, class_ids, class_ids, neighbors)
        results["robustness"]["k_sweep"][str(neighbors)] = {}
        for language in CONFIG["languages"]:
            metrics = evaluate_tied(minima[language].astype(float), tests, neighborhoods)
            compact = {"r2": metrics["r2"], "rmse": metrics["rmse"]}
            results["robustness"]["k_sweep"][str(neighbors)][language] = compact
            robustness_rows.append({"check": "k", "setting": str(neighbors),
                                    "language": language, **compact})

    scaling_values = {
        "function_weighted_zscore": scaled,
        "descriptor_weighted_zscore": StandardScaler().fit(
            unique_descriptors).transform(raw),
        "robust_median_iqr": RobustScaler().fit_transform(raw),
    }
    for scaling_name, values in scaling_values.items():
        tests, neighborhoods = tied_neighbor_plan(values, class_ids, class_ids, K)
        results["robustness"]["scaling"][scaling_name] = {}
        for language in CONFIG["languages"]:
            metrics = evaluate_tied(minima[language].astype(float), tests, neighborhoods)
            compact = {"r2": metrics["r2"], "rmse": metrics["rmse"]}
            results["robustness"]["scaling"][scaling_name][language] = compact
            robustness_rows.append({"check": "scaling", "setting": scaling_name,
                                    "language": language, **compact})
    results["elapsed_seconds"] = float(time.time() - started)

    predictions_frame.to_csv(artifacts / "predictions.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(artifacts / "protocol_metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(artifacts / "ablation_metrics.csv", index=False)
    pd.DataFrame(null_rows).to_csv(artifacts / "null_distributions.csv", index=False)
    pd.DataFrame(three_rows).to_csv(artifacts / "three_bit_tie_aware.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(artifacts / "robustness_metrics.csv", index=False)
    (artifacts / "analysis.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")

    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    protocol_frame = pd.DataFrame(prediction_rows)
    null_frame = pd.DataFrame(null_rows)
    ablation_frame = pd.DataFrame(ablation_rows)
    robustness_frame = pd.DataFrame(robustness_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    protocol_order = ["function_holdout", "orbit_holdout", "descriptor_holdout"]
    protocol_labels = ["function", "input orbit", "descriptor class"]
    x = np.arange(len(protocol_order))
    width = 0.23
    for offset, (language, color) in enumerate(zip(CONFIG["languages"], LANGUAGE_COLORS)):
        subset = protocol_frame[protocol_frame["language"] == language].set_index("protocol")
        axes[0].bar(x + (offset - 1) * width,
                    [subset.loc[name, "r2"] for name in protocol_order],
                    width=width, label=language.replace("_", "/"), color=color)
    axes[0].set_xticks(x, protocol_labels)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel(r"held-out $R^2$")
    axes[0].set_title("Prediction under stricter holdouts", loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16))

    null_values = [null_frame[null_frame["language"] == language]["null_r2"].to_numpy()
                   for language in CONFIG["languages"]]
    violin = axes[1].violinplot(null_values, positions=np.arange(3), widths=0.55,
                                showmeans=True, showextrema=False)
    for body, color in zip(violin["bodies"], LANGUAGE_COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    violin["cmeans"].set_color(CHARCOAL)
    observed = [results["languages"][language]["protocols"]
                ["descriptor_holdout"]["r2"] for language in CONFIG["languages"]]
    axes[1].scatter(np.arange(3), observed, marker="D", s=34,
                    c=LANGUAGE_COLORS, edgecolor="white", linewidth=0.5,
                    label="observed")
    axes[1].set_xticks(np.arange(3), [name.replace("_", "/")
                                      for name in CONFIG["languages"]])
    axes[1].set_ylim(0, 0.95)
    axes[1].set_ylabel(r"descriptor-held-out $R^2$")
    axes[1].set_title("Matched null versus observed", loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=2.2)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(figures / f"four_bit_validation.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65))
    ablation_order = ["bias_degree_symmetry", "influences_only", "fourier_only", "full"]
    ablation_labels = ["bias + degree\n+ symmetry", "influences", "Fourier", "full"]
    x = np.arange(len(ablation_order))
    for offset, (language, color) in enumerate(zip(CONFIG["languages"], LANGUAGE_COLORS)):
        subset = ablation_frame[ablation_frame["language"] == language].set_index("features")
        axes[0].bar(x + (offset - 1) * width,
                    [subset.loc[name, "r2"] for name in ablation_order],
                    width=width, label=language.replace("_", "/"), color=color)
    axes[0].axhline(0, color=CHARCOAL, linewidth=0.6)
    axes[0].set_xticks(x, ablation_labels, fontsize=8)
    axes[0].set_ylim(-0.12, 0.95)
    axes[0].set_ylabel(r"strict $R^2$")
    axes[0].set_title("Feature ablations", loc="left")
    axes[0].legend(frameon=False, fontsize=7)

    k_frame = robustness_frame[robustness_frame["check"] == "k"].copy()
    k_frame["k"] = k_frame["setting"].astype(int)
    for language, color in zip(CONFIG["languages"], LANGUAGE_COLORS):
        subset = k_frame[k_frame["language"] == language].sort_values("k")
        axes[1].plot(subset["k"], subset["r2"], marker="o", markersize=4,
                     color=color, label=language.replace("_", "/"))
    axes[1].set_xticks([3, 5, 7, 11, 15])
    axes[1].set_ylim(0.70, 0.90)
    axes[1].set_xlabel("neighbor threshold k")
    axes[1].set_ylabel(r"strict $R^2$")
    axes[1].set_title("Neighbor sensitivity", loc="left")

    scaling_order = ["function_weighted_zscore", "descriptor_weighted_zscore",
                     "robust_median_iqr"]
    scaling_labels = ["function-\nweighted z", "descriptor-\nweighted z", "median / IQR"]
    scaling_frame = robustness_frame[robustness_frame["check"] == "scaling"]
    x = np.arange(len(scaling_order))
    for offset, (language, color) in enumerate(zip(CONFIG["languages"], LANGUAGE_COLORS)):
        subset = scaling_frame[scaling_frame["language"] == language].set_index("setting")
        axes[2].bar(x + (offset - 1) * width,
                    [subset.loc[name, "r2"] for name in scaling_order],
                    width=width, color=color)
    axes[2].set_xticks(x, scaling_labels, fontsize=8)
    axes[2].set_ylim(0.55, 0.90)
    axes[2].set_ylabel(r"strict $R^2$")
    axes[2].set_title("Scaling sensitivity", loc="left")
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(figures / f"four_bit_robustness.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    lines = [
        "# Results: four-input semantic prediction of exact Boolean formula complexity", "",
        "## Design", "",
        f"All {FUNCTIONS:,} four-input Boolean functions were synthesized exactly in NAND, "
        "NOR, and AND/OR/NOT. The predictor uses only signed bias, sorted influences, "
        "Walsh--Fourier energy by degree, algebraic degree, and input-permutation "
        f"stabilizer size. It averages at least {K} nearest training functions, including "
        "the complete tied distance shell at the neighbor boundary. Complete test "
        "functions, input-permutation orbits, or exact descriptor classes are removed "
        "according to the protocol. The strict matched null permutes complexity within "
        "exact bias/degree/symmetry strata.", "",
        f"The universe contains {len(np.unique(orbit_groups)):,} input-permutation orbits "
        f"and {len(unique_descriptors):,} exact descriptor classes.", "",
        "## Main prediction result", "",
        "| Language | Function R² | Orbit R² | Descriptor R² | Descriptor ceiling R² | Strict RMSE | Null 95% interval | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language in CONFIG["languages"]:
        item = results["languages"][language]
        p = item["protocols"]
        null = item["matched_null"]
        lines.append(
            f"| {language.replace('_', '/')} | {p['function_holdout']['r2']:.3f} | "
            f"{p['orbit_holdout']['r2']:.3f} | {p['descriptor_holdout']['r2']:.3f} | "
            f"{item['descriptor_class_mean_ceiling_r2']:.3f} | "
            f"{p['descriptor_holdout']['rmse']:.3f} | "
            f"[{null['q025_r2']:.3f}, {null['q975_r2']:.3f}] | "
            f"{null['upper_tail_pvalue']:.4f} |")

    lines.extend(["", "## Exact synthesis scale", "",
                  "| Language | Maximum minimum gates | Mean minimum gates |",
                  "|---|---:|---:|"])
    for language in CONFIG["languages"]:
        item = results["languages"][language]
        lines.append(f"| {language.replace('_', '/')} | "
                     f"{item['maximum_minimum_gates']} | {item['mean_minimum_gates']:.3f} |")

    lines.extend(["", "## Descriptor ablations under strict holdout", "",
                  "| Language | Features | R² | RMSE |", "|---|---|---:|---:|"])
    for row in ablation_rows:
        lines.append(f"| {row['language'].replace('_', '/')} | {row['features']} | "
                     f"{row['r2']:.3f} | {row['rmse']:.3f} |")

    lines.extend(["", "## Cross-language rank stability", "",
                  "| Pair | Spearman rho |", "|---|---:|"])
    for pair, rho in results["cross_language_spearman"].items():
        lines.append(f"| {pair.replace('__', ' / ').replace('_', '/')} | {rho:.3f} |")

    lines.extend(["", "## Tie-aware three-bit comparison", "",
                  "| Language | Function R² | Orbit R² | Descriptor R² |",
                  "|---|---:|---:|---:|"])
    for language in CONFIG["languages"]:
        item = results["three_bit_tie_aware"][language]
        lines.append(f"| {language.replace('_', '/')} | "
                     f"{item['function_holdout']['r2']:.3f} | "
                     f"{item['orbit_holdout']['r2']:.3f} | "
                     f"{item['descriptor_holdout']['r2']:.3f} |")

    lines.extend(["", "## Robustness", "",
                  "### Neighbor threshold", "",
                  "| k | NAND R² | NOR R² | AND/OR/NOT R² |",
                  "|---:|---:|---:|---:|"])
    for neighbors in (3, 5, 7, 11, 15):
        item = results["robustness"]["k_sweep"][str(neighbors)]
        lines.append(f"| {neighbors} | {item['NAND']['r2']:.3f} | "
                     f"{item['NOR']['r2']:.3f} | {item['AND_OR_NOT']['r2']:.3f} |")
    lines.extend(["", "### Coordinate scaling", "",
                  "| Scaling | NAND R² | NOR R² | AND/OR/NOT R² |",
                  "|---|---:|---:|---:|"])
    for scaling_name in ("function_weighted_zscore", "descriptor_weighted_zscore",
                         "robust_median_iqr"):
        item = results["robustness"]["scaling"][scaling_name]
        lines.append(f"| {scaling_name} | {item['NAND']['r2']:.3f} | "
                     f"{item['NOR']['r2']:.3f} | {item['AND_OR_NOT']['r2']:.3f} |")

    lines.extend([
        "", "## Interpretation", "",
        "The three-bit result survives decisively over the complete four-input universe. "
        "Under this order-invariant tie-aware protocol, the strict R² values are numerically "
        "higher than in the earlier exact-seven experiment. "
        "After removing every function with the test descriptor, semantic neighborhoods explain "
        "77.7% of exact NAND/NOR variance and 84.3% of AND/OR/NOT variance. The corresponding "
        "matched-null intervals lie near 4--6%, and no permutation reaches an observed result "
        "(`p = 1/1001` in every language).", "",
        "The descriptor-class mean ceilings are 89.3% for NAND/NOR and 93.1% for AND/OR/NOT. "
        "Thus the strict out-of-class predictor recovers most, but not all, of the variance that "
        "this descriptor could possibly explain. Bias, degree, and symmetry alone explain "
        "essentially nothing under strict holdout, whereas influences alone and Fourier energy "
        "alone remain strongly predictive. Removing Fourier coordinates slightly improves the "
        "NAND/NOR kNN result; this indicates distance-metric dilution, not absence of Fourier "
        "signal, because Fourier-only prediction remains strong.", "",
        "Complexity is still representation-relative: NAND versus NOR rank correlation is 0.664, "
        "while each correlates 0.860 with AND/OR/NOT. This is a complete finite-universe result "
        "for four inputs, three related bases, a hand-designed descriptor, and an order-invariant "
        "tie-aware kNN rule. The tie policy is stricter and more stable than the exact-seven rule "
        "used in the current three-bit paper, so the raw R² values should not be read as a pure "
        "sample-size comparison. Exact targets pass both the three-bit regression and all-function "
        "NAND/NOR duality checks.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/semantic-boolean-complexity-4bit/scripts/run.py", "```", "",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
