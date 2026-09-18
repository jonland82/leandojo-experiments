"""Compare headline semantic isolation with matched declarations in the same repo."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
FRONTIER = ROOT / "experiments/frontier-formalizations"
STYLE = ROOT / "experiments/frontier-style-ablation"
VOCAB = ROOT / "experiments/frontier-vocabulary-ablation"
ARCH = ROOT / "experiments/frontier-architecture-controls"
GRAPH = ROOT / "experiments/frontier-dependency-shortcuts/artifacts/graph_nodes.jsonl"

FEATURES = [
    "statement_characters", "proof_characters", "file_lines", "file_declarations",
    "relative_declaration_position", "direct_imports", "scope_depth", "explicit_binders",
    "implicit_binders", "instance_binders", "arrow_forall_count", "delimiter_depth",
    "prior_definition_fraction", "local_references",
]
LOG_FEATURES = {"statement_characters", "proof_characters", "file_lines", "file_declarations",
                "direct_imports", "scope_depth", "explicit_binders", "implicit_binders",
                "instance_binders", "arrow_forall_count", "delimiter_depth", "local_references"}


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def density(queries: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    output = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        similarity = queries[start:start + 512] @ reference.T
        output[start:start + 512] = np.mean(np.partition(similarity, -k, axis=1)[:, -k:], axis=1)
    return output


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def headline_ancestor_source_ids(manifest: pd.DataFrame) -> set[int]:
    selected_by_location = {(row.project, row.file_path, int(row.line)): int(row.i)
                            for row in manifest.itertuples()}
    graph_nodes = jsonl(GRAPH)
    graph_by_location = {(row["project"], row["file_path"], int(row["line"])): row["id"]
                         for row in graph_nodes}
    source_by_graph = {graph_by_location[key]: source_i for key, source_i in selected_by_location.items()
                       if key in graph_by_location}
    headline_graph_ids = []
    for row in manifest[manifest["headline"]].itertuples():
        graph_id = graph_by_location.get((row.project, row.file_path, int(row.line)))
        if graph_id is None:
            candidates = [node["id"] for node in graph_nodes if node["project"] == row.project
                          and node["file_path"] == row.file_path
                          and node["name"].split(".")[-1] == row.name.split(".")[-1]
                          and abs(node["line"] - row.line) <= 5]
            if candidates:
                graph_id = candidates[0]
        if graph_id is not None:
            headline_graph_ids.append(graph_id)
    ancestors: set[int] = set()
    stack = list(headline_graph_ids)
    visited = set(stack)
    while stack:
        for dependency in graph_nodes[stack.pop()]["dependencies"]:
            if dependency not in visited:
                visited.add(dependency)
                ancestors.add(dependency)
                stack.append(dependency)
    return {source_by_graph[index] for index in ancestors if index in source_by_graph}


def match_specification(data: pd.DataFrame, excluded: set[int]) -> tuple[pd.DataFrame, dict]:
    rows = []
    imbalance_before, imbalance_after = [], []
    for project, project_data in data.groupby("project"):
        candidates = project_data[(~project_data["headline"]) &
                                  (~project_data["i"].isin(excluded))].copy()
        for headline in project_data[project_data["headline"]].itertuples():
            pool = candidates[candidates["cluster"] == headline.cluster].copy()
            if len(pool) < 1:
                raise RuntimeError(f"only {len(pool)} controls for {project}/{headline.name} cluster {headline.cluster}")
            matrix = []
            query = []
            for feature in FEATURES:
                values = pool[feature].to_numpy(dtype=float)
                value = float(getattr(headline, feature))
                if feature in LOG_FEATURES:
                    values, value = np.log1p(values), np.log1p(value)
                matrix.append(values)
                query.append(value)
            matrix = np.column_stack(matrix)
            query_array = np.asarray(query)[None, :]
            scaler = StandardScaler().fit(matrix)
            standardized = scaler.transform(matrix)
            standardized_query = scaler.transform(query_array)
            neighbor_count = min(CONFIG["matched_controls_per_headline"], len(pool))
            model = NearestNeighbors(n_neighbors=neighbor_count).fit(standardized)
            distances, indices = model.kneighbors(standardized_query)
            controls = pool.iloc[indices[0]]
            rows.append({"project": project, "headline": headline.name, "source_i": int(headline.i),
                         "controls": len(controls), "headline_density": float(headline.density),
                         "control_density": float(controls["density"].mean()),
                         "isolation_lift": float(controls["density"].mean() - headline.density),
                         "median_distance": float(np.median(distances[0])),
                         "maximum_distance": float(np.max(distances[0]))})
            pooled_sd = np.std(matrix, axis=0, ddof=1)
            variable = pooled_sd > 1e-8
            imbalance_before.extend(np.abs((np.asarray(query)[variable] - np.mean(
                matrix[:, variable], axis=0)) / pooled_sd[variable]))
            matched_matrix = matrix[indices[0]]
            imbalance_after.extend(np.abs((np.asarray(query)[variable] - np.mean(
                matched_matrix[:, variable], axis=0)) / pooled_sd[variable]))
    table = pd.DataFrame(rows)
    project_effects = table.groupby("project", as_index=False).agg(
        headlines=("headline", "count"), isolation_lift=("isolation_lift", "mean"),
        headline_density=("headline_density", "mean"), control_density=("control_density", "mean"))
    summary = {"headline_level": interval(table["isolation_lift"].to_numpy()),
               "project_balanced": interval(project_effects["isolation_lift"].to_numpy()),
               "mean_absolute_standardized_imbalance_before": float(np.mean(imbalance_before)),
               "mean_absolute_standardized_imbalance_after": float(np.mean(imbalance_after)),
               "minimum_controls_per_headline": int(table["controls"].min()),
               "median_controls_per_headline": float(table["controls"].median()),
               "projects": project_effects.to_dict(orient="records")}
    return table, summary


def make_figure(projects: pd.DataFrame) -> None:
    order = projects.sort_values("all_controls")["project"].tolist()
    plot = projects.set_index("project").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.scatter(plot["all_controls"], y - 0.12, color="#777777", label="all non-headlines")
    ax.scatter(plot["cone_excluded"], y + 0.12, color="#1d6996", label="headline cones excluded")
    for position, row in enumerate(plot.itertuples()):
        ax.plot([row.all_controls, row.cone_excluded], [position - 0.12, position + 0.12],
                color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [value.replace("navier-stokes-euler", "Navier–Stokes") for value in order])
    ax.set_xlabel("headline isolation lift over same-repository controls")
    ax.set_title("Are the headline theorems themselves unusual?", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "headline_isolation.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "headline_isolation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source_manifest = pd.DataFrame(jsonl(FRONTIER / "inputs/manifest.jsonl"))
    source_manifest = source_manifest[source_manifest["project"].isin(CONFIG["primary_projects"])].copy()
    architecture = pd.DataFrame(jsonl(ARCH / "artifacts/features.jsonl"))
    architecture = architecture[architecture["cohort"] == "frontier"].copy()
    vocab_manifest = pd.DataFrame(jsonl(VOCAB / "inputs/manifest.jsonl"))
    vocab_embeddings = normalize(np.load(
        VOCAB / "artifacts/embeddings/vocabulary_anonymized.npy", allow_pickle=False))
    style_manifest = pd.DataFrame(jsonl(STYLE / "inputs/manifest.jsonl"))
    style_embeddings = normalize(np.load(
        STYLE / "artifacts/embeddings/alpha_normalized.npy", allow_pickle=False))
    reference = style_embeddings[style_manifest["cohort"].to_numpy() == "reference"]
    vocab_density = density(vocab_embeddings, reference, CONFIG["neighbor_k"])
    density_table = vocab_manifest[["source_i"]].rename(columns={"source_i": "i"})
    density_table["density"] = vocab_density
    clusters = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)["statement_cluster"]
    density_table["cluster"] = clusters[vocab_manifest["source_i"].to_numpy(dtype=int)]
    data = source_manifest.merge(density_table, on="i", validate="one_to_one").merge(
        architecture, left_on="i", right_on="source_i", validate="one_to_one", suffixes=("", "_arch"))
    ancestors = headline_ancestor_source_ids(source_manifest)
    all_table, all_result = match_specification(data, set())
    strict_table, strict_result = match_specification(data, ancestors)
    result = {"experiment_id": CONFIG["experiment_id"], "headlines": int(data["headline"].sum()),
              "sampled_headline_ancestors": len(ancestors), "all_nonheadline_controls": all_result,
              "headline_cone_excluded_controls": strict_result,
              "interpretation": "Positive lift means headline statements are less dense than architecture-matched declarations in the same repository and raw semantic cluster.",
              "limitations": ["only 34 modern headline theorems across eight project-level replication units",
                              "headline selection was supplied by repository metadata or the experiment manifest",
                              "project-vocabulary resolution is conservative and lexical"]}
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    all_table.to_csv(artifacts / "headline_matches_all.csv", index=False)
    strict_table.to_csv(artifacts / "headline_matches_cone_excluded.csv", index=False)
    all_projects = pd.DataFrame(all_result["projects"])[["project", "isolation_lift"]].rename(
        columns={"isolation_lift": "all_controls"})
    strict_projects = pd.DataFrame(strict_result["projects"])[["project", "isolation_lift"]].rename(
        columns={"isolation_lift": "cone_excluded"})
    projects = all_projects.merge(strict_projects, on="project", validate="one_to_one")
    non_flt = projects[projects["project"] != "flt"]
    sensitivity = {"all_nonheadline_controls": interval(non_flt["all_controls"].to_numpy()),
                   "headline_cone_excluded_controls": interval(non_flt["cone_excluded"].to_numpy())}
    result["sensitivity_excluding_macro_obscured_flt"] = sensitivity
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    projects.to_csv(artifacts / "project_effects.csv", index=False)
    make_figure(projects)
    primary, strict = all_result["project_balanced"], strict_result["project_balanced"]
    lines = ["# Results: within-repository headline isolation", "", "## Main result", "",
        f"Across 34 modern headline theorems, the project-balanced vocabulary-anonymized isolation lift over "
        f"same-repository, same-cluster architecture-matched declarations is **{primary['mean']:.4f}** "
        f"(95% project-level t interval {primary['lower']:.4f} to {primary['upper']:.4f}).", "",
        f"After excluding sampled declarations inside any headline prerequisite cone, the lift is "
        f"**{strict['mean']:.4f}** ({strict['lower']:.4f} to {strict['upper']:.4f}).", "",
        f"Excluding FLT's macro-wrapped endpoints, the corresponding strict lift is "
        f"**{sensitivity['headline_cone_excluded_controls']['mean']:.4f}** "
        f"({sensitivity['headline_cone_excluded_controls']['lower']:.4f} to "
        f"{sensitivity['headline_cone_excluded_controls']['upper']:.4f}).", "",
        "## Project effects", "", "| Project | All controls | Cones excluded |", "|---|---:|---:|"]
    for row in projects.itertuples():
        lines.append(f"| {row.project} | {row.all_controls:.4f} | {row.cone_excluded:.4f} |")
    lines.extend(["", "![Headline isolation](figures/headline_isolation.png)", "", "## Interpretation", "",
        "If the interval contains zero, the earlier frontier separation is a repository-level distribution "
        "shift rather than a property demonstrated specifically by the named major theorems. A positive lift "
        "would instead motivate a concept-combination novelty analysis focused on the headlines."])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"headlines": int(data["headline"].sum()), "all": primary,
                      "cone_excluded": strict, "sampled_ancestors": len(ancestors)}, indent=2))


if __name__ == "__main__":
    main()
