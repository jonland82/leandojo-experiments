"""Relate dependency synthesis depth to alpha-normalized semantic isolation."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
STYLE = ROOT / "experiments/frontier-style-ablation"
TIME = ROOT / "experiments/frontier-time-controls"
FRONTIER = ROOT / "experiments/frontier-formalizations"
REFERENCE = ROOT / "experiments/semantic-embeddings-10000"
ARCHITECTURE = ROOT / "experiments/frontier-architecture-controls"
DEPENDENCY = ROOT / "experiments/frontier-dependency-shortcuts"
TIME_CONFIG = json.loads((TIME / "config.json").read_text(encoding="utf-8"))

ARCHITECTURE_COLUMNS = [
    "statement_characters", "proof_characters", "file_lines", "file_declarations",
    "relative_declaration_position", "direct_imports", "scope_depth", "explicit_binders",
    "implicit_binders", "instance_binders", "arrow_forall_count", "delimiter_depth",
    "prior_definition_fraction", "local_references",
]


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def topk_density(queries: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    density = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 256):
        end = min(len(queries), start + 256)
        similarity = queries[start:end] @ reference.T
        density[start:end] = np.mean(np.partition(similarity, -k, axis=1)[:, -k:], axis=1)
    return density


def raw_clusters() -> tuple[np.ndarray, np.ndarray]:
    centers = normalize(np.load(REFERENCE / "artifacts/clusters/statement_centers.npy", allow_pickle=False))
    controls = normalize(np.load(TIME / "artifacts/embeddings/statement.npy", allow_pickle=False))
    mapping = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)
    return (np.argmax(controls @ centers.T, axis=1).astype(np.int16),
            mapping["statement_cluster"].astype(np.int16))


def isolation_residuals() -> pd.DataFrame:
    manifest = pd.DataFrame(jsonl(STYLE / "inputs/manifest.jsonl"))
    inputs = jsonl(STYLE / "inputs/alpha_normalized.jsonl")
    lengths = np.asarray([row["statement_characters"] for row in inputs])
    embeddings = normalize(np.load(
        STYLE / "artifacts/embeddings/alpha_normalized.npy", allow_pickle=False))
    reference_mask = manifest["cohort"].to_numpy() == "reference"
    control_mask = manifest["cohort"].to_numpy() == "control"
    frontier_mask = manifest["cohort"].to_numpy() == "frontier"
    density = topk_density(embeddings[~reference_mask], embeddings[reference_mask], CONFIG["neighbor_k"])
    control_n = int(control_mask.sum())
    control_clusters, frontier_clusters = raw_clusters()
    controls = manifest.loc[control_mask, ["revision", "name"]].reset_index(drop=True)
    controls["length"] = lengths[control_mask]
    controls["density"] = density[:control_n]
    controls["cluster"] = control_clusters
    frontier = manifest.loc[frontier_mask, ["source_i", "project", "name"]].reset_index(drop=True)
    frontier = frontier.rename(columns={"source_i": "i"})
    source_locations = pd.DataFrame(jsonl(FRONTIER / "inputs/manifest.jsonl"))[
        ["i", "file_path", "line"]]
    frontier = frontier.merge(source_locations, on="i", validate="one_to_one")
    frontier["length"] = lengths[frontier_mask]
    frontier["density"] = density[control_n:]
    frontier["cluster"] = frontier_clusters
    frontier["isolation_residual"] = np.nan
    for project, revision in TIME_CONFIG["projects"].items():
        front_positions = np.flatnonzero(frontier["project"].to_numpy() == project)
        control_project = controls.loc[controls["revision"] == revision].reset_index(drop=True)
        scale = float(np.std(np.log1p(control_project["length"]), ddof=1))
        for cluster in np.unique(frontier.iloc[front_positions]["cluster"]):
            fp = front_positions[frontier.iloc[front_positions]["cluster"].to_numpy() == cluster]
            cp = np.flatnonzero(control_project["cluster"].to_numpy() == cluster)
            model = NearestNeighbors(n_neighbors=1).fit(
                np.log1p(control_project.iloc[cp]["length"].to_numpy())[:, None])
            query = np.log1p(frontier.iloc[fp]["length"].to_numpy())
            _, nearest = model.kneighbors(query[:, None])
            matched_density = control_project.iloc[cp[nearest[:, 0]]]["density"].to_numpy()
            frontier.loc[fp, "isolation_residual"] = matched_density - frontier.iloc[fp]["density"].to_numpy()
    frontier = frontier[frontier["project"].isin(TIME_CONFIG["projects"])].copy()
    if frontier["isolation_residual"].isna().any():
        raise RuntimeError("unmatched frontier isolation residuals")
    return frontier


def graph_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    nodes = jsonl(DEPENDENCY / "artifacts/graph_nodes.jsonl")
    dependents = np.zeros(len(nodes), dtype=np.int32)
    location: dict[tuple[str, str, int], int] = {}
    names: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for node in nodes:
        location[(node["project"], node["file_path"], node["line"])] = node["id"]
        names[(node["project"], node["file_path"], node["name"].split(".")[-1])].append(node["id"])
        for dependency in node["dependencies"]:
            dependents[dependency] += 1
    rows = []
    for number, row in enumerate(selected.itertuples(), 1):
        index = location.get((row.project, row.file_path, row.line))
        if index is None:
            candidates = names.get((row.project, row.file_path, row.name.split(".")[-1]), [])
            nearby = [value for value in candidates if abs(nodes[value]["line"] - row.line) <= 5]
            exact = [value for value in nearby if nodes[value]["name"] == row.name]
            pool = exact or nearby
            index = min(pool, key=lambda value: abs(nodes[value]["line"] - row.line)) if pool else None
        if index is None:
            raise RuntimeError(f"graph node not found: {row.project} {row.file_path}:{row.line} {row.name}")
        distances = {index: 0}
        queue = deque([index])
        while queue:
            current = queue.popleft()
            for dependency in nodes[current]["dependencies"]:
                if dependency not in distances:
                    distances[dependency] = distances[current] + 1
                    queue.append(dependency)
        cone = set(distances) - {index}
        savings = 0.0
        total_cost = 0.0
        for dependency in cone:
            reuse = int(dependents[dependency])
            body = nodes[dependency]["proof_characters"]
            savings += max(0, (reuse - 1) * body - reuse * len(nodes[dependency]["full_name"]))
            total_cost += body
        rows.append({"i": row.i, "graph_id": index, "dependency_depth": max(distances.values()),
                     "closure_size": len(cone), "direct_dependencies": len(nodes[index]["dependencies"]),
                     "direct_dependents": int(dependents[index]),
                     "compression_fraction": savings / max(1.0, savings + total_cost)})
        if number % 1000 == 0:
            print(f"graph metrics: {number:,}/{len(selected):,}", flush=True)
    return pd.DataFrame(rows)


def predictor_matrix(group: pd.DataFrame, include_structure: bool) -> np.ndarray:
    base = []
    for column in ARCHITECTURE_COLUMNS:
        values = group[column].to_numpy(dtype=float)
        if column in {"statement_characters", "proof_characters", "file_lines", "file_declarations",
                      "direct_imports", "scope_depth", "explicit_binders", "implicit_binders",
                      "instance_binders", "arrow_forall_count", "delimiter_depth", "local_references"}:
            values = np.log1p(values)
        base.append(values)
    depth = np.log1p(group["dependency_depth"].to_numpy(dtype=float))
    compression = group["compression_fraction"].to_numpy(dtype=float)
    matrix = np.column_stack(base)
    if include_structure:
        matrix = np.column_stack([matrix, depth, compression])
    return matrix


def standardized_ols(group: pd.DataFrame, include_structure: bool) -> tuple[float, float]:
    y = group["isolation_residual"].to_numpy(dtype=float)
    matrix = predictor_matrix(group, include_structure)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)
    keep = scales > 1e-12
    matrix = (matrix[:, keep] - means[keep]) / scales[keep]
    design = np.column_stack([np.ones(len(group)), matrix])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    if not include_structure:
        return float("nan"), float("nan")
    original_indices = np.flatnonzero(keep)
    base_count = len(ARCHITECTURE_COLUMNS)
    depth_position = int(np.flatnonzero(original_indices == base_count)[0]) + 1
    compression_position = int(np.flatnonzero(original_indices == base_count + 1)[0]) + 1
    return float(coefficients[depth_position]), float(coefficients[compression_position])


def cross_validated_r2(group: pd.DataFrame, include_structure: bool) -> float:
    y = group["isolation_residual"].to_numpy(dtype=float)
    matrix = predictor_matrix(group, include_structure)
    prediction = np.empty(len(group), dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=20260918)
    for train, test in splitter.split(matrix):
        means = matrix[train].mean(axis=0)
        scales = matrix[train].std(axis=0, ddof=1)
        keep = scales > 1e-12
        train_x = (matrix[train][:, keep] - means[keep]) / scales[keep]
        test_x = (matrix[test][:, keep] - means[keep]) / scales[keep]
        design = np.column_stack([np.ones(len(train)), train_x])
        coefficients, _, _, _ = np.linalg.lstsq(design, y[train], rcond=None)
        prediction[test] = np.column_stack([np.ones(len(test)), test_x]) @ coefficients
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def interval(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def make_figure(projects: pd.DataFrame) -> None:
    order = projects.sort_values("depth_coefficient")["project"].tolist()
    table = projects.set_index("project").loc[order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
    axes[0].barh(y, table["depth_coefficient"], color="#1d6996")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, [value.replace("navier-stokes-euler", "Navier–Stokes") for value in order])
    axes[0].set_xlabel("isolation change per SD log depth")
    axes[0].set_title("Partial depth association", loc="left")
    axes[1].barh(y, 100 * table["incremental_r2"], color="#edae49")
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("incremental explained variance (%)")
    axes[1].set_title("Depth + compression contribution", loc="left")
    fig.suptitle("Does synthesis depth explain semantic isolation?", x=0.08, ha="left")
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "depth_isolation.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "depth_isolation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metric_cache = artifacts / "declaration_metrics.jsonl"
    if metric_cache.exists():
        data = pd.DataFrame(jsonl(metric_cache))
        print(f"loaded {len(data):,} cached declaration metrics", flush=True)
    else:
        isolation = isolation_residuals()
        isolation = isolation[isolation["project"].isin(CONFIG["primary_projects"])].copy()
        metrics = graph_metrics(isolation)
        architecture = pd.DataFrame(jsonl(ARCHITECTURE / "artifacts/features.jsonl"))
        architecture = architecture[architecture["cohort"] == "frontier"].copy()
        data = isolation.merge(metrics, on="i", validate="one_to_one").merge(
            architecture, left_on="i", right_on="source_i", validate="one_to_one", suffixes=("", "_arch"))
    projects = []
    for project, group in data.groupby("project"):
        depth, compression = standardized_ols(group, True)
        base_r2 = cross_validated_r2(group, False)
        full_r2 = cross_validated_r2(group, True)
        projects.append({"project": project, "n": len(group), "depth_coefficient": depth,
                         "compression_coefficient": compression, "base_r2": base_r2,
                         "full_r2": full_r2, "incremental_r2": full_r2 - base_r2,
                         "mean_isolation": float(group["isolation_residual"].mean())})
    project_table = pd.DataFrame(projects)
    depth_result = interval(project_table["depth_coefficient"].to_numpy())
    compression_result = interval(project_table["compression_coefficient"].to_numpy())
    incremental = interval(project_table["incremental_r2"].to_numpy())
    mean_isolation = float(project_table["mean_isolation"].mean())
    depth_gap_fraction = depth_result["mean"] / mean_isolation
    result = {"experiment_id": CONFIG["experiment_id"], "declarations": len(data),
              "project_balanced_depth_coefficient": depth_result,
              "project_balanced_compression_coefficient": compression_result,
              "project_balanced_incremental_r2": incremental,
              "project_balanced_mean_isolation": mean_isolation,
              "depth_coefficient_fraction_of_mean_isolation": depth_gap_fraction,
              "projects": projects,
              "interpretation": "Coefficients are within-project partial associations with positive exact-snapshot isolation residual; they are not causal effects.",
              "limitations": ["dependency edges are lexical rather than elaborated",
                              "depth is maximum shortest-path prerequisite distance",
                              "project intervals have eight replication units"]}
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    project_table.to_csv(artifacts / "project_effects.csv", index=False)
    with metric_cache.open("w", encoding="utf-8", newline="\n") as stream:
        for row in data.to_dict(orient="records"):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_figure(project_table)
    lines = ["# Results: synthesis depth and semantic isolation", "", "## Main result", "",
        f"Across eight modern projects, one within-project SD of log dependency depth is associated with "
        f"**{depth_result['mean']:.4f}** higher alpha-normalized isolation residual (95% project-level t "
        f"interval {depth_result['lower']:.4f} to {depth_result['upper']:.4f}) after the prior architecture "
        f"controls. The point estimate is **{100 * depth_gap_fraction:.1f}%** of the mean residual frontier "
        "gap per SD of depth, but is not distinguishable from zero across projects.", "",
        f"The corresponding compression coefficient is **{compression_result['mean']:.4f}** "
        f"({compression_result['lower']:.4f} to {compression_result['upper']:.4f}). Together, depth and "
        f"compression add **{100 * incremental['mean']:.2f}%** five-fold out-of-sample explained variance on average "
        f"({100 * incremental['lower']:.2f}% to {100 * incremental['upper']:.2f}%).", "",
        "## Project effects", "", "| Project | n | Depth coefficient | Compression coefficient | Incremental R² |",
        "|---|---:|---:|---:|---:|"]
    for row in project_table.itertuples():
        lines.append(f"| {row.project} | {row.n} | {row.depth_coefficient:.4f} | "
                     f"{row.compression_coefficient:.4f} | {row.incremental_r2:.4f} |")
    lines.extend(["", "![Depth and isolation](figures/depth_isolation.png)", "", "## Interpretation", "",
        "The outcome already removes exact-snapshot, coarse semantic-cluster, and statement-length effects. "
        "The regression additionally controls observed source architecture. A positive depth coefficient "
        "has the predicted sign, but its project-level interval crosses zero and therefore does not establish "
        "depth as an explanation of isolation. Lexical graph omissions—especially proof macros—remain the "
        "main measurement limitation."])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"declarations": len(data), "depth": depth_result,
                      "compression": compression_result, "incremental_r2": incremental}, indent=2))


if __name__ == "__main__":
    main()
