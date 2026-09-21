"""Cross-validated permutation importance for the expanded semantic descriptor."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


EXPERIMENT = Path(__file__).resolve().parents[1]
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"
LANGUAGES = ["NAND", "NOR", "AND_OR_NOT"]
FOLDS = 5
REPEATS = 8
SEED = 0


def class_targets(target: np.ndarray, groups: np.ndarray, classes: int):
    weights = np.bincount(groups, minlength=classes).astype(float)
    sums = np.bincount(groups, weights=target, minlength=classes)
    square_sums = np.bincount(groups, weights=target ** 2, minlength=classes)
    means = sums / weights
    within_sse = square_sums - sums ** 2 / weights
    return weights, means, within_sse


def function_weighted_r2(prediction: np.ndarray, weights: np.ndarray,
                         means: np.ndarray, within_sse: np.ndarray) -> float:
    overall_mean = float(np.average(means, weights=weights))
    residual = float(np.sum(within_sse + weights * (means - prediction) ** 2))
    total = float(np.sum(within_sse + weights * (means - overall_mean) ** 2))
    return 1 - residual / total


def main() -> None:
    table = pd.read_csv(ARTIFACTS / "feature_table.csv")
    definitions = pd.read_csv(ARTIFACTS / "feature_definitions.csv")
    feature_names = definitions.feature.tolist()
    values = table[feature_names].to_numpy(dtype=float)
    unique_values, representative, groups = np.unique(
        values, axis=0, return_index=True, return_inverse=True)
    classes = len(unique_values)
    definitions = definitions.set_index("feature").loc[feature_names].reset_index()
    family_columns = {
        family: np.flatnonzero(definitions.family.to_numpy() == family)
        for family in definitions.family.unique()
    }
    status = definitions.set_index("feature").status.to_dict()
    family = definitions.set_index("feature").family.to_dict()

    fold_rows = []
    individual_rows = []
    family_rows = []
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for language_index, language in enumerate(LANGUAGES):
        target = table[f"target_{language}"].to_numpy(dtype=float)
        weights, means, within_sse = class_targets(target, groups, classes)
        for fold, (train, test) in enumerate(splitter.split(unique_values)):
            model = HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=SEED + 10 * language_index + fold,
            )
            model.fit(unique_values[train], means[train], sample_weight=weights[train])
            test_values = unique_values[test]
            baseline_prediction = model.predict(test_values)
            baseline = function_weighted_r2(
                baseline_prediction, weights[test], means[test], within_sse[test])
            fold_rows.append({"language": language, "fold": fold,
                              "test_descriptor_classes": len(test), "r2": baseline})
            rng = np.random.default_rng(SEED + 100 * language_index + fold)
            permutations = [rng.permutation(len(test)) for _ in range(REPEATS)]

            for column, name in enumerate(feature_names):
                for repeat, permutation in enumerate(permutations):
                    permuted = test_values.copy()
                    permuted[:, column] = test_values[permutation, column]
                    score = function_weighted_r2(
                        model.predict(permuted), weights[test], means[test], within_sse[test])
                    individual_rows.append({
                        "language": language, "fold": fold, "repeat": repeat,
                        "feature": name, "status": status[name], "family": family[name],
                        "permutation_importance": baseline - score,
                    })

            for family_name, columns in family_columns.items():
                for repeat, permutation in enumerate(permutations):
                    permuted = test_values.copy()
                    permuted[:, columns] = test_values[permutation][:, columns]
                    score = function_weighted_r2(
                        model.predict(permuted), weights[test], means[test], within_sse[test])
                    family_rows.append({
                        "language": language, "fold": fold, "repeat": repeat,
                        "family": family_name,
                        "features": len(columns),
                        "permutation_importance": baseline - score,
                    })

    folds = pd.DataFrame(fold_rows)
    individual_raw = pd.DataFrame(individual_rows)
    family_raw = pd.DataFrame(family_rows)
    folds.to_csv(ARTIFACTS / "importance_model_folds.csv", index=False)

    individual = (individual_raw.groupby(
        ["feature", "status", "family", "language"], as_index=False)
        .permutation_importance.agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"}))
    family_importance = (family_raw.groupby(
        ["family", "features", "language"], as_index=False)
        .permutation_importance.agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"}))
    individual.to_csv(ARTIFACTS / "feature_permutation_importance.csv", index=False)
    family_importance.to_csv(ARTIFACTS / "feature_family_permutation_importance.csv", index=False)

    individual_ranking = (individual.groupby(["feature", "status", "family"], as_index=False)
                          .importance_mean.mean()
                          .sort_values("importance_mean", ascending=False))
    family_ranking = (family_importance.groupby(["family", "features"], as_index=False)
                      .importance_mean.mean()
                      .sort_values("importance_mean", ascending=False))
    individual_ranking.to_csv(ARTIFACTS / "feature_importance_ranking.csv", index=False)
    family_ranking.to_csv(ARTIFACTS / "feature_family_importance_ranking.csv", index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    top_count = 30
    selected_features = individual_ranking.head(top_count).iloc[::-1]
    selected_families = family_ranking.iloc[::-1]
    language_colors = {"NAND": "#40566A", "NOR": "#87939C", "AND_OR_NOT": "#25292C"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [1.45, 1]})

    names = selected_features.feature.tolist()
    colors = ["#B98249" if selected_features.set_index("feature").loc[name, "status"] == "new"
              else "#7A8C99" for name in names]
    axes[0].barh(np.arange(len(names)), selected_features.importance_mean, color=colors, alpha=0.55)
    offsets = {"NAND": -0.18, "NOR": 0.0, "AND_OR_NOT": 0.18}
    indexed = individual.set_index(["feature", "language"])
    for language in LANGUAGES:
        axes[0].scatter([indexed.loc[(name, language), "importance_mean"] for name in names],
                        np.arange(len(names)) + offsets[language], color=language_colors[language],
                        s=20, label=language)
    axes[0].set_yticks(np.arange(len(names)),
                       [f"{'NEW' if status[name] == 'new' else 'USED'}  {name}" for name in names],
                       fontsize=8)
    axes[0].set_title("Individual conditional importance", loc="left", fontweight="bold")
    axes[0].set_xlabel("held-out R2 decrease after permutation")
    axes[0].legend(frameon=False, loc="lower right")

    family_names = selected_families.family.tolist()
    axes[1].barh(np.arange(len(family_names)), selected_families.importance_mean,
                 color="#B98249", alpha=0.55)
    family_indexed = family_importance.set_index(["family", "language"])
    for language in LANGUAGES:
        axes[1].scatter(
            [family_indexed.loc[(name, language), "importance_mean"] for name in family_names],
            np.arange(len(family_names)) + offsets[language], color=language_colors[language], s=20)
    axes[1].set_yticks(np.arange(len(family_names)),
                       [f"{name} ({selected_families.set_index('family').loc[name, 'features']})"
                        for name in family_names], fontsize=8)
    axes[1].set_title("Joint family importance", loc="left", fontweight="bold")
    axes[1].set_xlabel("held-out R2 decrease after joint permutation")
    for axis in axes:
        axis.axvline(0, color="#25292C", linewidth=0.8)
        axis.grid(axis="x", color="#D9DDDF", linewidth=0.7)
        axis.set_axisbelow(True)
    fig.suptitle("Which semantic features the nonlinear regressor uses", x=0.06,
                 ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"feature_permutation_importance.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    fold_summary = (folds.groupby("language", as_index=False).r2.agg(["mean", "std"])
                    .reset_index().rename(columns={"mean": "r2_mean", "std": "r2_std"}))
    top_individual = individual_ranking.head(30)
    top_family = family_ranking
    lines = [
        "# Conditional feature importance", "",
        "A nonlinear histogram-gradient-boosting regressor was evaluated with five-fold",
        "holdout of complete 66-feature descriptor classes. Each class is represented",
        "once during fitting, weighted by its number of Boolean functions; evaluation",
        "restores the irreducible within-class error. Importance is the decrease in",
        "held-out R2 after permuting a feature across descriptor classes, averaged over",
        f"{FOLDS} folds and {REPEATS} permutations per fold.", "",
        "Individual importance can be suppressed by correlated substitutes. The family",
        "analysis therefore permutes every coordinate in a conceptual family jointly.",
        "Negative values mean the permutation marginally improved generalization and",
        "should be interpreted as zero importance, not as a beneficial feature.", "",
        "![Permutation importance](figures/feature_permutation_importance.png)", "",
        "## Held-out regression performance", "",
        fold_summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Individual ranking", "",
        top_individual.to_markdown(index=False, floatfmt=".4f"), "",
        "## Joint family ranking", "",
        top_family.to_markdown(index=False, floatfmt=".4f"), "",
        "## Interpretation rule", "",
        "- High standalone association but low permutation importance means the feature",
        "  is largely redundant with other coordinates.",
        "- High individual importance means the fitted regression function specifically",
        "  depends on that coordinate.",
        "- High family importance means the underlying kind of structure matters even",
        "  when its individual coordinates substitute for one another.", "",
        "This is predictive importance, not a causal or lower-bound theorem.", "",
        "## Reproduce", "", "```powershell",
        "python experiments/semantic-boolean-complexity-4bit/scripts/analyze_feature_importance.py",
        "```", "",
    ]
    (EXPERIMENT / "FEATURE_IMPORTANCE.md").write_text("\n".join(lines), encoding="utf-8")
    result = {
        "descriptor_classes": classes,
        "folds": FOLDS,
        "permutation_repeats": REPEATS,
        "top_individual_features": individual_ranking.head(10).feature.tolist(),
        "top_feature_families": family_ranking.head(10).family.tolist(),
        "mean_r2": dict(zip(fold_summary.language, fold_summary.r2_mean)),
    }
    (ARTIFACTS / "feature_importance_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
