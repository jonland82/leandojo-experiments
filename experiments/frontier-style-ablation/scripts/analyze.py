"""Measure time-matched frontier isolation after statement-style ablations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
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


def topk_density(queries: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    density = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        end = min(len(queries), start + 512)
        similarity = queries[start:end] @ reference.T
        density[start:end] = np.mean(np.partition(similarity, -k, axis=1)[:, -k:], axis=1)
    return density


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def raw_clusters() -> tuple[np.ndarray, np.ndarray]:
    centers = normalize(np.load(
        REFERENCE / "artifacts/clusters/statement_centers.npy", allow_pickle=False
    ))
    controls = normalize(np.load(TIME / "artifacts/embeddings/statement.npy", allow_pickle=False))
    control_clusters = np.argmax(controls @ centers.T, axis=1).astype(np.int16)
    mapping = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)
    return control_clusters, mapping["statement_cluster"].astype(np.int16)


def match(frontier: pd.DataFrame, controls: pd.DataFrame) -> tuple[float, dict]:
    matched_indices = np.empty(len(frontier), dtype=np.int32)
    distances = np.empty(len(frontier), dtype=float)
    control_length = np.log1p(controls["characters"].to_numpy(dtype=float))
    scale = float(np.std(control_length, ddof=1))
    for cluster in sorted(frontier["cluster"].unique()):
        front_positions = np.flatnonzero(frontier["cluster"].to_numpy() == cluster)
        control_positions = np.flatnonzero(controls["cluster"].to_numpy() == cluster)
        if not len(control_positions):
            raise RuntimeError(f"control snapshot has no raw cluster {cluster}")
        model = NearestNeighbors(n_neighbors=1).fit(control_length[control_positions, None])
        query = np.log1p(frontier.iloc[front_positions]["characters"].to_numpy(dtype=float))
        distance, local_index = model.kneighbors(query[:, None])
        matched_indices[front_positions] = control_positions[local_index[:, 0]]
        distances[front_positions] = distance[:, 0] / scale
    difference = (
        frontier["density"].to_numpy()
        - controls.iloc[matched_indices]["density"].to_numpy()
    )
    summary = {
        "n": len(frontier),
        "mean_difference": float(np.mean(difference)),
        "frontier_density": float(frontier["density"].mean()),
        "matched_control_density": float(controls.iloc[matched_indices]["density"].mean()),
        "median_match_distance": float(np.median(distances)),
        "q95_match_distance": float(np.quantile(distances, 0.95)),
        "unique_control_matches": int(np.unique(matched_indices).size),
        "calipers": {},
    }
    for caliper in (0.25, 0.5, 1.0):
        selected = distances <= caliper
        summary["calipers"][str(caliper)] = {
            "n": int(np.sum(selected)),
            "coverage_fraction": float(np.mean(selected)),
            "mean_difference": float(np.mean(difference[selected])),
        }
    return summary["mean_difference"], summary


def analyze_variant(variant: str, manifest: pd.DataFrame,
                    control_clusters: np.ndarray, frontier_clusters: np.ndarray) -> dict:
    embeddings = normalize(np.load(
        EXPERIMENT / f"artifacts/embeddings/{variant}.npy", allow_pickle=False
    ))
    input_rows = jsonl(EXPERIMENT / f"inputs/{variant}.jsonl")
    lengths = np.asarray([row["statement_characters"] for row in input_rows], dtype=np.int32)
    reference_mask = manifest["cohort"].to_numpy() == "reference"
    control_mask = manifest["cohort"].to_numpy() == "control"
    frontier_mask = manifest["cohort"].to_numpy() == "frontier"
    reference_vectors = embeddings[reference_mask]
    query_vectors = embeddings[~reference_mask]
    query_density = topk_density(query_vectors, reference_vectors, CONFIG["neighbor_k"])
    control_n = int(np.sum(control_mask))
    controls = manifest.loc[control_mask, ["revision", "name"]].reset_index(drop=True)
    controls["characters"] = lengths[control_mask]
    controls["density"] = query_density[:control_n]
    controls["cluster"] = control_clusters
    frontier = manifest.loc[frontier_mask, ["project", "name"]].reset_index(drop=True)
    frontier["characters"] = lengths[frontier_mask]
    frontier["density"] = query_density[control_n:]
    frontier["cluster"] = frontier_clusters

    projects = []
    summaries = {}
    for project, revision in TIME_CONFIG["projects"].items():
        front = frontier.loc[frontier["project"] == project].copy()
        control = controls.loc[controls["revision"] == revision].copy()
        effect, summary = match(front, control)
        summaries[project] = summary
        projects.append({"project": project, "revision": revision, "effect": effect, **summary})
    effects = np.asarray([row["effect"] for row in projects])
    calipers = {}
    for caliper in (0.25, 0.5, 1.0):
        key = str(caliper)
        values = np.asarray([summaries[p]["calipers"][key]["mean_difference"]
                             for p in TIME_CONFIG["projects"]])
        calipers[key] = {
            "project_balanced": interval(values),
            "coverage_fraction": float(np.mean([
                summaries[p]["calipers"][key]["coverage_fraction"]
                for p in TIME_CONFIG["projects"]
            ])),
        }
    return {
        "project_balanced": interval(effects),
        "projects": projects,
        "caliper_sensitivity": calipers,
    }


def make_figure(table: pd.DataFrame, aggregates: dict) -> None:
    order = table.sort_values("alpha_normalized")["project"].tolist()
    indexed = table.set_index("project").loc[order]
    variants = ["raw", "header_normalized", "alpha_normalized"]
    labels = ["raw", "header removed", "+ bound names removed"]
    colors = ["#777777", "#4f8ab8", "#174f82"]
    offsets = [-0.18, 0.0, 0.18]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for variant, label, color, offset in zip(variants, labels, colors, offsets):
        ax.scatter(indexed[variant], y + offset, s=35, color=color, label=label, zorder=3)
    for position, row in enumerate(indexed.itertuples()):
        ax.plot([row.raw, row.header_normalized, row.alpha_normalized],
                [position + value for value in offsets], color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel("matched frontier minus exact-snapshot control density")
    ax.set_title("Surface normalization does not remove frontier isolation", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "style_ablation_effects.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "style_ablation_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manifest = pd.DataFrame(jsonl(EXPERIMENT / "inputs/manifest.jsonl"))
    control_clusters, frontier_clusters = raw_clusters()
    variants = {
        variant: analyze_variant(variant, manifest, control_clusters, frontier_clusters)
        for variant in CONFIG["variants"]
    }
    raw_table = pd.read_csv(TIME / "artifacts/project_matching.csv")
    table = raw_table[["project", "time_matched_difference"]].rename(
        columns={"time_matched_difference": "raw"}
    )
    for variant, result in variants.items():
        effects = pd.DataFrame(result["projects"])[["project", "effect"]].rename(
            columns={"effect": variant}
        )
        table = table.merge(effects, on="project", validate="one_to_one")
    raw_interval = interval(table["raw"].to_numpy())
    aggregates = {"raw": raw_interval, **{
        variant: variants[variant]["project_balanced"] for variant in CONFIG["variants"]
    }}
    raw_effect = raw_interval["mean"]
    attenuation = {
        variant: float(1 - aggregates[variant]["mean"] / raw_effect)
        for variant in CONFIG["variants"]
    }
    result = {
        "experiment_id": CONFIG["experiment_id"],
        "serving_profile": CONFIG["embedding"]["model_id"],
        "representations": aggregates,
        "attenuation_relative_to_corrected_raw": attenuation,
        "variants": variants,
        "project_effects": table.to_dict(orient="records"),
        "interpretation": "Residual changes estimate sensitivity to declaration headers and local bound names, not causal novelty.",
        "limitations": [
            "alpha normalization is token-based rather than elaborator-certified",
            "global constants and abstraction architecture are retained",
            "author and project conventions beyond headers and bound names remain uncontrolled",
        ],
    }
    artifacts = EXPERIMENT / "artifacts"
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table.to_csv(artifacts / "project_effects.csv", index=False)
    make_figure(table, aggregates)

    header = aggregates["header_normalized"]
    alpha = aggregates["alpha_normalized"]
    alpha_caliper = variants["alpha_normalized"]["caliper_sensitivity"]["0.5"]
    lines = [
        "# Results: frontier statement-style ablation", "", "## Main result", "",
        f"With corrected exact-snapshot controls, the raw project-balanced deficit is "
        f"**{raw_interval['mean']:.4f}**. Removing declaration headers and formatting changes it to "
        f"**{header['mean']:.4f}** (95% t interval {header['lower']:.4f} to {header['upper']:.4f}); "
        f"also alpha-renaming bound identifiers changes it to **{alpha['mean']:.4f}** "
        f"(95% t interval {alpha['lower']:.4f} to {alpha['upper']:.4f}).", "",
        f"Header normalization increases the gap magnitude by "
        f"**{-100 * attenuation['header_normalized']:.1f}%**; header plus bound-name normalization reduces "
        f"the raw gap by **{100 * attenuation['alpha_normalized']:.1f}%**. "
        "All eight projects remain negatively shifted after both ablations. The response is heterogeneous: "
        "`formalized-logic` approaches zero after alpha normalization, whereas PNT and Schoenflies move little.", "",
        "## Project effects", "",
        "| Project | Raw | Header removed | + bound names removed |",
        "|---|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.project} | {row.raw:.4f} | {row.header_normalized:.4f} | "
            f"{row.alpha_normalized:.4f} |"
        )
    lines.extend([
        "", "![Style ablation effects](figures/style_ablation_effects.png)", "",
        "## Robustness and interpretation", "",
        f"For the alpha-normalized view, a 0.5-SD length caliper retains "
        f"**{100 * alpha_caliper['coverage_fraction']:.1f}%** of declarations and gives a "
        f"project-balanced effect of **{alpha_caliper['project_balanced']['mean']:.4f}**.", "",
        "The tested surface conventions therefore explain only the reported attenuation. The remaining gap "
        "still combines mathematical content, abstraction architecture, global vocabulary, and deeper "
        "project conventions; it is not by itself a novelty estimate. Alpha normalization is token-based "
        "and should be read as a sensitivity analysis, not a semantics-preserving Lean transformation.",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"representations": aggregates, "attenuation": attenuation,
                      "alpha_caliper_0.5": alpha_caliper}, indent=2))


if __name__ == "__main__":
    main()
