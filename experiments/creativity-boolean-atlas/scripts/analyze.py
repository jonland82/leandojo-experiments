"""Build a semantic atlas with exact synthesis in three Boolean languages."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


NAVY = "#40566A"
SLATE = "#87939C"
LIGHT_GRAY = "#D9DDDF"
CHARCOAL = "#25292C"
MUTED_CMAP = LinearSegmentedColormap.from_list(
    "muted_navy", ["#F7F7F4", LIGHT_GRAY, SLATE, NAVY, CHARCOAL])
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": CHARCOAL, "axes.labelcolor": CHARCOAL,
                     "text.color": CHARCOAL, "xtick.color": CHARCOAL,
                     "ytick.color": CHARCOAL, "savefig.facecolor": "white"})


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
N = CONFIG["variables"]
MASK = (1 << (1 << N)) - 1


def variable_table(index: int) -> int:
    return sum(((assignment >> index) & 1) << assignment
               for assignment in range(1 << N))


VARIABLES = [variable_table(index) for index in range(N)]


def operation(language: str, left: int, right: int | None = None) -> int:
    if language == "NAND":
        return (~(left & int(right))) & MASK
    if language == "NOR":
        return (~(left | int(right))) & MASK
    if right is None:
        return (~left) & MASK
    raise ValueError("mixed binary operation must be selected explicitly")


def synthesize(language: str) -> tuple[pd.DataFrame, list[int]]:
    counts: list[dict[int, int]] = [{table: 1 for table in VARIABLES}]
    totals = [N]
    minima = {table: 0 for table in VARIABLES}
    for size in range(1, CONFIG["maximum_gates"] + 1):
        current: dict[int, int] = {}
        if language == "AND_OR_NOT":
            for table, multiplicity in counts[size - 1].items():
                output = (~table) & MASK
                current[output] = current.get(output, 0) + multiplicity
        for left_size in range(size):
            right_size = size - 1 - left_size
            for left, left_count in counts[left_size].items():
                for right, right_count in counts[right_size].items():
                    multiplicity = left_count * right_count
                    if language == "AND_OR_NOT":
                        outputs = (left & right, left | right)
                    else:
                        outputs = (operation(language, left, right),)
                    for output in outputs:
                        current[output] = current.get(output, 0) + multiplicity
        counts.append(current)
        totals.append(sum(current.values()))
        if language in {"NAND", "NOR"}:
            catalan = math.comb(2 * size, size) // (size + 1)
            if totals[-1] != catalan * N ** (size + 1):
                raise RuntimeError(f"{language} syntax-count invariant failed at {size}")
        else:
            expected = totals[size - 1] + 2 * sum(
                totals[left] * totals[size - 1 - left] for left in range(size))
            if totals[-1] != expected:
                raise RuntimeError(f"{language} syntax-count invariant failed at {size}")
        for table in current:
            minima.setdefault(table, size)
    if len(minima) != 256:
        raise RuntimeError(f"{language} synthesized only {len(minima)} functions")
    rows = []
    for table in range(256):
        size = minima[table]
        boundary_count = counts[size][table]
        probability = boundary_count / totals[size]
        rows.append({"truth_table": table, "language": language,
                     "minimum_gates": size, "boundary_formula_count": boundary_count,
                     "all_formulas_at_boundary": totals[size],
                     "boundary_probability": probability,
                     "boundary_surprisal_bits": -math.log2(probability)})
    return pd.DataFrame(rows), totals


def combine_cost(costs: list[int], binary_cost) -> int:
    if len(costs) == 1:
        return costs[0]
    midpoint = len(costs) // 2
    return binary_cost(combine_cost(costs[:midpoint], binary_cost),
                       combine_cost(costs[midpoint:], binary_cost))


def normal_cost(language: str, table: int, form: str) -> int:
    """Cost of canonical DNF or CNF in a language-specific native direction."""
    selected = [a for a in range(1 << N)
                if bool((table >> a) & 1) == (form == "dnf")]
    if not selected:
        return 2 if language == "AND_OR_NOT" else 3
    if language == "AND_OR_NOT":
        binary = lambda left, right: 1 + left + right
        outer = binary
        terms = []
        for assignment in selected:
            if form == "dnf":
                literals = [0 if (assignment >> i) & 1 else 1 for i in range(N)]
            else:
                literals = [1 if (assignment >> i) & 1 else 0 for i in range(N)]
            terms.append(combine_cost(literals, binary))
        return combine_cost(terms, outer)
    binary = lambda left, right: 3 + 2 * left + 2 * right
    literal_costs = []
    for assignment in selected:
        if form == "dnf":
            literals = [0 if (assignment >> i) & 1 else 1 for i in range(N)]
        else:
            literals = [1 if (assignment >> i) & 1 else 0 for i in range(N)]
        literal_costs.append(combine_cost(literals, binary))
    return combine_cost(literal_costs, binary)


def baseline_cost(language: str, table: int) -> int:
    complement = (~table) & MASK
    if language == "NAND":
        return min(normal_cost(language, table, "dnf"),
                   1 + 2 * normal_cost(language, complement, "dnf"))
    if language == "NOR":
        return min(normal_cost(language, table, "cnf"),
                   1 + 2 * normal_cost(language, complement, "cnf"))
    candidates = [normal_cost(language, table, "dnf"),
                  normal_cost(language, table, "cnf"),
                  1 + normal_cost(language, complement, "dnf"),
                  1 + normal_cost(language, complement, "cnf")]
    return min(candidates)


def walsh(values: np.ndarray) -> np.ndarray:
    result = values.astype(float).copy()
    stride = 1
    while stride < len(result):
        for start in range(0, len(result), 2 * stride):
            for offset in range(stride):
                a = result[start + offset]
                b = result[start + offset + stride]
                result[start + offset] = a + b
                result[start + offset + stride] = a - b
        stride *= 2
    return result / len(result)


def semantic_features(table: int) -> dict[str, float]:
    bits = np.array([(table >> a) & 1 for a in range(1 << N)], dtype=int)
    signs = 2 * bits - 1
    coefficients = walsh(signs)
    energy = [sum(coefficients[m] ** 2 for m in range(1 << N)
                  if m.bit_count() == degree) for degree in range(N + 1)]
    influences = []
    for variable in range(N):
        changes = sum(bits[a] != bits[a ^ (1 << variable)] for a in range(1 << N))
        influences.append(changes / (1 << N))
    anf = bits.copy()
    for variable in range(N):
        for mask in range(1 << N):
            if mask & (1 << variable):
                anf[mask] ^= anf[mask ^ (1 << variable)]
    degree = max((mask.bit_count() for mask, value in enumerate(anf) if value), default=0)
    symmetries = 0
    for permutation in itertools.permutations(range(N)):
        invariant = True
        for assignment in range(1 << N):
            mapped = sum(((assignment >> permutation[i]) & 1) << i for i in range(N))
            if bits[assignment] != bits[mapped]:
                invariant = False
                break
        symmetries += int(invariant)
    ordered_influences = sorted(influences)
    result = {"truth_table": table, "bias": float(signs.mean()),
              "algebraic_degree": degree, "input_symmetries": symmetries}
    result.update({f"influence_{i}": value for i, value in enumerate(ordered_influences)})
    result.update({f"fourier_energy_degree_{i}": value for i, value in enumerate(energy)})
    return result


def semantic_knn_prediction(features: np.ndarray, target: np.ndarray, neighbors: int = 7) -> np.ndarray:
    predictions = []
    for index in range(len(target)):
        distances = np.linalg.norm(features - features[index], axis=1)
        nearest = np.argsort(distances + np.eye(len(target))[index] * 1e9)[:neighbors]
        predictions.append(float(target[nearest].mean()))
    return np.asarray(predictions)


def main() -> None:
    semantic = pd.DataFrame([semantic_features(table) for table in range(256)])
    feature_columns = [column for column in semantic if column != "truth_table"]
    scaled = StandardScaler().fit_transform(semantic[feature_columns])
    coordinates = PCA(n_components=2, random_state=CONFIG["seed"]).fit_transform(scaled)
    semantic["semantic_pc1"] = coordinates[:, 0]
    semantic["semantic_pc2"] = coordinates[:, 1]

    frames = []
    syntax_totals = {}
    prediction_results = {}
    for language in CONFIG["languages"]:
        frame, totals = synthesize(language)
        frame["canonical_baseline_gates"] = [baseline_cost(language, table)
                                              for table in frame["truth_table"]]
        frame["compression_gain"] = (frame["canonical_baseline_gates"]
                                      - frame["minimum_gates"])
        target = frame["minimum_gates"].to_numpy(dtype=float)
        prediction = semantic_knn_prediction(scaled, target)
        prediction_results[language] = {
            "semantic_knn_r2": float(r2_score(target, prediction)),
            "semantic_knn_rmse": float(mean_squared_error(target, prediction) ** 0.5),
            "mean_baseline_rmse": float(np.sqrt(np.mean((target - target.mean()) ** 2))),
        }
        frames.append(frame.merge(semantic, on="truth_table"))
        syntax_totals[language] = totals
    atlas = pd.concat(frames, ignore_index=True)

    stability = {}
    for metric in ("minimum_gates", "boundary_surprisal_bits", "compression_gain"):
        matrix = atlas.pivot(index="truth_table", columns="language", values=metric)
        stability[metric] = {}
        for left, right in itertools.combinations(CONFIG["languages"], 2):
            correlation = stats.spearmanr(matrix[left], matrix[right])
            stability[metric][f"{left}__{right}"] = {
                "rho": float(correlation.statistic), "pvalue": float(correlation.pvalue)}

    targets = {
        "and3": 128,
        "majority": sum((int(((a >> 0) & 1) + ((a >> 1) & 1) + ((a >> 2) & 1) >= 2)) << a
                        for a in range(8)),
        "parity": sum(((a.bit_count() % 2) == 1) << a for a in range(8)),
        "mux": sum(((((a >> 1) & 1) if ((a >> 0) & 1) else ((a >> 2) & 1))) << a
                   for a in range(8)),
    }
    artifacts = EXPERIMENT / "artifacts"
    figures = EXPERIMENT / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(artifacts / "boolean_atlas.csv", index=False)
    semantic.to_csv(artifacts / "semantic_coordinates.csv", index=False)
    result = {"experiment_id": CONFIG["experiment_id"],
              "functions": 256, "languages": CONFIG["languages"],
              "semantic_invariant_classes": int(semantic[feature_columns]
                                                   .drop_duplicates().shape[0]),
              "distinct_projected_positions": int(semantic[["semantic_pc1", "semantic_pc2"]]
                                                     .drop_duplicates().shape[0]),
              "syntax_count_invariants_verified": True,
              "semantic_knn_prediction": prediction_results,
              "cross_language_rank_stability": stability,
              "targets": {name: int(table) for name, table in targets.items()}}
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), sharex=True, sharey=True)
    maximum_size = atlas["minimum_gates"].max()
    for axis, language in zip(axes, CONFIG["languages"]):
        subset = atlas[atlas["language"] == language]
        classes = subset.groupby(["semantic_pc1", "semantic_pc2"], as_index=False).agg(
            minimum_gates=("minimum_gates", "mean"), class_size=("truth_table", "size"))
        scatter = axis.scatter(classes["semantic_pc1"], classes["semantic_pc2"],
                               c=classes["minimum_gates"], cmap=MUTED_CMAP,
                               vmin=0, vmax=maximum_size,
                               s=18 + 8 * np.sqrt(classes["class_size"]),
                               alpha=0.85, edgecolors="white", linewidths=0.3)
        axis.set_title(language.replace("_", "/"), loc="left")
        axis.set_xlabel("semantic PC1")
        for name, table in targets.items():
            row = subset[subset["truth_table"] == table].iloc[0]
            axis.annotate(name, (row.semantic_pc1, row.semantic_pc2), xytext=(3, 3),
                          textcoords="offset points", fontsize=7)
    axes[0].set_ylabel("semantic PC2")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("exact minimum gates")
    fig.suptitle("One semantic map, three formula languages", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.14, top=0.84, wspace=0.08)
    fig.savefig(figures / "boolean_semantic_atlas.png", dpi=220)
    fig.savefig(figures / "boolean_semantic_atlas.pdf")
    plt.close(fig)

    min_stability = stability["minimum_gates"]
    lines = ["# Results: Boolean semantic atlas", "",
             "All 256 functions were synthesized exactly in NAND, NOR, and AND/OR/NOT. "
             "The same syntax-free semantic coordinates are used in every panel. The 36 exact "
             "invariant-descriptor classes occupy 22 visible positions in the two-dimensional "
             "projection; point area indicates projected multiplicity and color is mean exact "
             "minimum size.", "",
             "## Cross-language tests", "",
             "| Metric | Language pair | Spearman rho |", "|---|---|---:|"]
    for metric, pairs in stability.items():
        for pair, values in pairs.items():
            lines.append(f"| {metric.replace('_', ' ')} | {pair.replace('__', ' / ')} | "
                         f"{values['rho']:.3f} |")
    lines.extend(["", "## Semantic prediction", "",
                  "A leave-one-function-out seven-neighbor predictor uses only the semantic descriptors.", "",
                  "| Language | Semantic kNN R² | RMSE | Mean-only RMSE |",
                  "|---|---:|---:|---:|"])
    for language, values in prediction_results.items():
        lines.append(f"| {language} | {values['semantic_knn_r2']:.3f} | "
                     f"{values['semantic_knn_rmse']:.3f} | {values['mean_baseline_rmse']:.3f} |")
    lines.extend(["", "## Interpretation", "",
                  "Persistence of a ranking across languages is evidence for a semantic component; "
                  "instability is evidence that the chosen syntax prior matters. Compression remains "
                  "baseline-relative in every language. The atlas therefore tests representation "
                  "robustness rather than assuming it.", "",
                  "![Boolean semantic atlas](figures/boolean_semantic_atlas.png)"])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
