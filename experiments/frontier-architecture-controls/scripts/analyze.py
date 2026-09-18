"""Adjust alpha-normalized frontier isolation for static source architecture."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
STYLE = ROOT / "experiments/frontier-style-ablation"
TIME = ROOT / "experiments/frontier-time-controls"
FRONTIER = ROOT / "experiments/frontier-formalizations"
REFERENCE = ROOT / "experiments/semantic-embeddings-10000"
TIME_CONFIG = json.loads((TIME / "config.json").read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def topk_density(queries: np.ndarray, reference: np.ndarray, k: int = 10) -> np.ndarray:
    output = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        end = min(len(queries), start + 512)
        similarity = queries[start:end] @ reference.T
        output[start:end] = np.mean(np.partition(similarity, -k, axis=1)[:, -k:], axis=1)
    return output


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def add_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for source in (
        "statement_characters", "proof_characters", "file_lines", "file_declarations",
        "direct_imports", "local_references",
    ):
        frame[f"log_{source}"] = np.log1p(frame[source].astype(float))
    return frame


def attach_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    embeddings = normalize(np.load(
        STYLE / "artifacts/embeddings/alpha_normalized.npy", allow_pickle=False
    ))
    reference = embeddings[:10000]
    controls = features[features["cohort"] == "control"].copy().sort_values("source_i")
    frontier = features[features["cohort"] == "frontier"].copy().sort_values("source_i")
    control_vectors = embeddings[10000 + controls["source_i"].to_numpy(dtype=int)]
    frontier_vectors = embeddings[31000 + frontier["source_i"].to_numpy(dtype=int)]
    controls["density"] = topk_density(control_vectors, reference)
    frontier["density"] = topk_density(frontier_vectors, reference)

    centers = normalize(np.load(
        REFERENCE / "artifacts/clusters/statement_centers.npy", allow_pickle=False
    ))
    raw_controls = normalize(np.load(TIME / "artifacts/embeddings/statement.npy", allow_pickle=False))
    all_control_clusters = np.argmax(raw_controls @ centers.T, axis=1).astype(int)
    controls["cluster"] = all_control_clusters[controls["source_i"].to_numpy(dtype=int)]
    mapping = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)
    frontier["cluster"] = mapping["statement_cluster"][frontier["source_i"].to_numpy(dtype=int)]
    return add_transforms(pd.concat([controls, frontier], ignore_index=True))


def prepare_project(frontier: pd.DataFrame, controls: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    feature_names = CONFIG["features"]
    control_x = controls[feature_names].to_numpy(dtype=float)
    frontier_x = frontier[feature_names].to_numpy(dtype=float)
    mean = np.mean(control_x, axis=0)
    scale = np.std(control_x, axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    control_z = np.clip((control_x - mean) / scale, -10, 10)
    frontier_z = np.clip((frontier_x - mean) / scale, -10, 10)
    maximum = min(CONFIG["matching"]["maximum_components"], control_z.shape[1])
    full = PCA(n_components=maximum, random_state=0).fit(control_z)
    target = CONFIG["matching"]["pca_variance"]
    components = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), target) + 1)
    components = min(maximum, components)
    control_pca = full.transform(control_z)[:, :components]
    frontier_pca = full.transform(frontier_z)[:, :components]
    diagnostic = {
        "components": components,
        "explained_variance": float(np.sum(full.explained_variance_ratio_[:components])),
        "before_mean_absolute_smd": float(np.mean(np.abs(np.mean(frontier_z, axis=0)))),
    }
    return control_pca, frontier_pca, diagnostic


def match_project(frontier: pd.DataFrame, controls: pd.DataFrame) -> tuple[dict, np.ndarray]:
    control_pca, frontier_pca, diagnostic = prepare_project(frontier, controls)
    indices = np.empty(len(frontier), dtype=np.int32)
    distances = np.empty(len(frontier), dtype=float)
    for cluster in sorted(frontier["cluster"].unique()):
        front_positions = np.flatnonzero(frontier["cluster"].to_numpy() == cluster)
        control_positions = np.flatnonzero(controls["cluster"].to_numpy() == cluster)
        if not len(control_positions):
            raise RuntimeError(f"no architecture controls in cluster {cluster}")
        model = NearestNeighbors(n_neighbors=1).fit(control_pca[control_positions])
        distance, local = model.kneighbors(frontier_pca[front_positions])
        indices[front_positions] = control_positions[local[:, 0]]
        distances[front_positions] = distance[:, 0]
    difference = frontier["density"].to_numpy() - controls.iloc[indices]["density"].to_numpy()

    feature_names = CONFIG["features"]
    control_x = controls[feature_names].to_numpy(dtype=float)
    frontier_x = frontier[feature_names].to_numpy(dtype=float)
    mean = np.mean(control_x, axis=0)
    scale = np.std(control_x, axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    matched_smd = np.mean((frontier_x - controls.iloc[indices][feature_names].to_numpy()) / scale, axis=0)
    diagnostic.update({
        "mean_difference": float(np.mean(difference)),
        "frontier_density": float(frontier["density"].mean()),
        "matched_control_density": float(controls.iloc[indices]["density"].mean()),
        "median_distance": float(np.median(distances)),
        "q95_distance": float(np.quantile(distances, 0.95)),
        "unique_control_matches": int(np.unique(indices).size),
        "after_mean_absolute_smd": float(np.mean(np.abs(matched_smd))),
        "feature_smd_after": dict(zip(feature_names, map(float, matched_smd))),
        "distance_calipers": {},
    })
    for caliper in (1.0, 2.0, 3.0):
        selected = distances <= caliper
        diagnostic["distance_calipers"][str(caliper)] = {
            "n": int(np.sum(selected)),
            "coverage_fraction": float(np.mean(selected)),
            "mean_difference": float(np.mean(difference[selected])) if np.any(selected) else None,
        }
    return diagnostic, indices


def regression_effect(frontier: pd.DataFrame, controls: pd.DataFrame) -> dict:
    feature_names = CONFIG["features"]
    combined = pd.concat([controls.copy(), frontier.copy()], ignore_index=True)
    combined["frontier_indicator"] = np.r_[np.zeros(len(controls)), np.ones(len(frontier))]
    for feature in feature_names:
        mean = float(controls[feature].mean())
        scale = float(controls[feature].std(ddof=1))
        combined[f"z_{feature}"] = (combined[feature] - mean) / (scale if scale > 1e-8 else 1.0)
    formula = "density ~ frontier_indicator + " + " + ".join(
        f"z_{feature}" for feature in feature_names
    ) + " + C(cluster)"
    model = smf.ols(formula, data=combined).fit(cov_type="HC3")
    lower, upper = model.conf_int().loc["frontier_indicator"]
    return {
        "effect": float(model.params["frontier_indicator"]),
        "standard_error": float(model.bse["frontier_indicator"]),
        "lower": float(lower),
        "upper": float(upper),
        "r_squared": float(model.rsquared),
    }


def nonlinear_effect(frontier: pd.DataFrame, controls: pd.DataFrame) -> dict:
    """Control-trained nonlinear outcome adjustment with OOF control residuals."""
    feature_names = CONFIG["features"]
    control_x = controls[feature_names].to_numpy(dtype=float)
    frontier_x = frontier[feature_names].to_numpy(dtype=float)
    mean = np.mean(control_x, axis=0)
    scale = np.std(control_x, axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    control_x = np.clip((control_x - mean) / scale, -10, 10)
    frontier_x = np.clip((frontier_x - mean) / scale, -10, 10)
    cluster_count = int(max(controls["cluster"].max(), frontier["cluster"].max()) + 1)
    control_x = np.c_[control_x, np.eye(cluster_count)[controls["cluster"].to_numpy(dtype=int)]]
    frontier_x = np.c_[frontier_x, np.eye(cluster_count)[frontier["cluster"].to_numpy(dtype=int)]]
    control_y = controls["density"].to_numpy(dtype=float)
    frontier_y = frontier["density"].to_numpy(dtype=float)
    parameters = dict(
        n_estimators=200, min_samples_leaf=10, max_features=0.7,
        random_state=0, n_jobs=1,
    )
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    control_prediction = cross_val_predict(
        RandomForestRegressor(**parameters), control_x, control_y, cv=folds, n_jobs=-1
    )
    model = RandomForestRegressor(**{**parameters, "n_jobs": -1}).fit(control_x, control_y)
    frontier_prediction = model.predict(frontier_x)
    control_residual = control_y - control_prediction
    frontier_residual = frontier_y - frontier_prediction
    return {
        "effect": float(np.mean(frontier_residual) - np.mean(control_residual)),
        "control_cv_r_squared": float(r2_score(control_y, control_prediction)),
        "mean_frontier_prediction": float(np.mean(frontier_prediction)),
        "mean_control_prediction": float(np.mean(control_prediction)),
    }


def make_figure(table: pd.DataFrame) -> None:
    order = table.sort_values("architecture_nonlinear")["project"].tolist()
    indexed = table.set_index("project").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.scatter(indexed["alpha_baseline"], y - 0.18, color="#777777", s=36,
               label="length/domain/time/style controls")
    ax.scatter(indexed["architecture_regression"], y, color="#a05a91", s=36,
               label="+ architecture (linear)")
    ax.scatter(indexed["architecture_nonlinear"], y + 0.18, color="#6b245c", s=36,
               label="+ architecture (nonlinear)")
    for position, row in enumerate(indexed.itertuples()):
        ax.plot([row.alpha_baseline, row.architecture_regression, row.architecture_nonlinear],
                [position - 0.18, position, position + 0.18], color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel("matched frontier minus exact-snapshot control density")
    ax.set_title("Static architecture adjustment leaves a residual gap", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "architecture_adjustment.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "architecture_adjustment.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    features = pd.DataFrame(jsonl(EXPERIMENT / "artifacts/features.jsonl"))
    frame = attach_outcomes(features)
    baseline = pd.read_csv(STYLE / "artifacts/project_effects.csv")[
        ["project", "alpha_normalized"]
    ].rename(columns={"alpha_normalized": "alpha_baseline"})
    project_rows = []
    details = {}
    regression_rows = []
    nonlinear_rows = []
    for project, revision in TIME_CONFIG["projects"].items():
        frontier = frame[(frame["cohort"] == "frontier") & (frame["project"] == project)].copy()
        controls = frame[(frame["cohort"] == "control") & (frame["revision"] == revision)].copy()
        matched, _ = match_project(frontier, controls)
        regression = regression_effect(frontier, controls)
        nonlinear = nonlinear_effect(frontier, controls)
        details[project] = matched
        project_rows.append({"project": project, "architecture_matched": matched["mean_difference"]})
        regression_rows.append({"project": project, "architecture_regression": regression["effect"]})
        nonlinear_rows.append({
            "project": project,
            "architecture_nonlinear": nonlinear["effect"],
            "control_cv_r_squared": nonlinear["control_cv_r_squared"],
        })
    table = baseline.merge(pd.DataFrame(project_rows), on="project", validate="one_to_one")
    table = table.merge(pd.DataFrame(regression_rows), on="project", validate="one_to_one")
    table = table.merge(pd.DataFrame(nonlinear_rows), on="project", validate="one_to_one")
    baseline_interval = interval(table["alpha_baseline"].to_numpy())
    matched_interval = interval(table["architecture_matched"].to_numpy())
    regression_interval = interval(table["architecture_regression"].to_numpy())
    nonlinear_interval = interval(table["architecture_nonlinear"].to_numpy())
    attenuation = {
        "matching": float(1 - matched_interval["mean"] / baseline_interval["mean"]),
        "linear": float(1 - regression_interval["mean"] / baseline_interval["mean"]),
        "nonlinear": float(1 - nonlinear_interval["mean"] / baseline_interval["mean"]),
    }
    balance_before = float(np.mean([value["before_mean_absolute_smd"] for value in details.values()]))
    balance_after = float(np.mean([value["after_mean_absolute_smd"] for value in details.values()]))
    result = {
        "experiment_id": CONFIG["experiment_id"],
        "baseline_alpha_normalized": baseline_interval,
        "architecture_matched": matched_interval,
        "architecture_regression": regression_interval,
        "architecture_nonlinear": nonlinear_interval,
        "attenuation_by_estimator": attenuation,
        "mean_absolute_standardized_difference": {
            "before": balance_before, "after": balance_after,
        },
        "project_effects": table.to_dict(orient="records"),
        "project_matching": details,
        "features": CONFIG["features"],
        "limitations": [
            "source-static architecture is not an elaborated dependency graph",
            "local reference reuse is lexical and approximate",
            "matching remains observational and cannot identify mathematical novelty causally",
        ],
    }
    artifacts = EXPERIMENT / "artifacts"
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table.to_csv(artifacts / "project_effects.csv", index=False)
    make_figure(table)

    lines = [
        "# Results: static abstraction-architecture controls", "", "## Main result", "",
        f"The alpha-normalized project-balanced deficit is **{baseline_interval['mean']:.4f}**. "
        f"Within-project fixed-cluster linear adjustment gives **{regression_interval['mean']:.4f}** "
        f"(project-level 95% t interval {regression_interval['lower']:.4f} to "
        f"{regression_interval['upper']:.4f}); a cross-fitted nonlinear adjustment gives "
        f"**{nonlinear_interval['mean']:.4f}** (95% t interval {nonlinear_interval['lower']:.4f} to "
        f"{nonlinear_interval['upper']:.4f}). These represent attenuations of "
        f"**{100 * attenuation['linear']:.1f}%** and **{100 * attenuation['nonlinear']:.1f}%**, respectively.", "",
        "Thus the apparent explanatory share is model-sensitive: only the linear specification yields material "
        "attenuation, while the nonlinear control-trained model leaves the gap essentially unchanged.", "",
        f"Direct architecture matching gives **{matched_interval['mean']:.4f}**, but it is not the preferred "
        "estimate because several frontier projects have weak common support in the control architecture space.", "",
        f"Mean absolute standardized feature imbalance falls from **{balance_before:.3f}** before matching "
        f"to **{balance_after:.3f}** afterward.", "",
        "## Project effects", "",
        "| Project | Alpha baseline | Direct match | Linear adjustment | Nonlinear adjustment |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.project} | {row.alpha_baseline:.4f} | {row.architecture_matched:.4f} | "
            f"{row.architecture_regression:.4f} | {row.architecture_nonlinear:.4f} |"
        )
    lines.extend([
        "", "![Architecture adjustment](figures/architecture_adjustment.png)", "",
        "## Interpretation", "",
        "The adjustment measures observable source organization, not formal dependency geometry. A persistent "
        "negative effect is evidence against simple module/binder/proof-size architecture as the whole "
        "explanation, but the residual still mixes global mathematical vocabulary, elaborated dependencies, "
        "author choices, and mathematical novelty. The next discriminating test is an actual dependency-shortcut "
        "analysis around major lemmas.",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline_interval,
        "architecture_matched": matched_interval,
        "architecture_regression": regression_interval,
        "architecture_nonlinear": nonlinear_interval,
        "attenuation": attenuation,
        "balance_before": balance_before,
        "balance_after": balance_after,
    }, indent=2))


if __name__ == "__main__":
    main()
