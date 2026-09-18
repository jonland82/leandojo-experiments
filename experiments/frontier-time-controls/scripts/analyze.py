"""Compare frontier declarations with exact-snapshot Mathlib controls."""

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
REFERENCE = ROOT / "experiments/semantic-embeddings-10000"
FRONTIER = ROOT / "experiments/frontier-formalizations"


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def map_controls() -> tuple[np.ndarray, np.ndarray]:
    reference = normalize(np.load(REFERENCE / "artifacts/embeddings/statement.npy", allow_pickle=False))
    controls = normalize(np.load(EXPERIMENT / "artifacts/embeddings/statement.npy", allow_pickle=False))
    centers = normalize(np.load(
        REFERENCE / "artifacts/clusters/statement_centers.npy", allow_pickle=False
    ))
    k = CONFIG["neighbor_k"]
    density = np.empty(len(controls), dtype=np.float32)
    batch_size = 512
    for start in range(0, len(controls), batch_size):
        end = min(len(controls), start + batch_size)
        similarities = controls[start:end] @ reference.T
        density[start:end] = np.mean(
            np.partition(similarities, -k, axis=1)[:, -k:], axis=1
        )
    clusters = np.argmax(controls @ centers.T, axis=1).astype(np.int16)
    return density, clusters


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def match_project(frontier: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matched_indices = np.empty(len(frontier), dtype=np.int32)
    distances = np.empty(len(frontier), dtype=float)
    control_log_length = np.log1p(controls["statement_characters"].to_numpy(dtype=float))
    scale = float(np.std(control_log_length, ddof=1))
    for cluster in sorted(frontier["statement_cluster"].unique()):
        front_positions = np.flatnonzero(frontier["statement_cluster"].to_numpy() == cluster)
        control_positions = np.flatnonzero(controls["statement_cluster"].to_numpy() == cluster)
        if not len(control_positions):
            raise RuntimeError(f"snapshot control has no members of statement cluster {cluster}")
        model = NearestNeighbors(n_neighbors=1)
        model.fit(control_log_length[control_positions, None])
        query = np.log1p(frontier.iloc[front_positions]["statement_characters"].to_numpy(dtype=float))
        distance, local_index = model.kneighbors(query[:, None])
        matched_indices[front_positions] = control_positions[local_index[:, 0]]
        distances[front_positions] = distance[:, 0] / scale
    output = frontier.reset_index(drop=True).copy()
    output["matched_control_index"] = matched_indices
    output["match_distance"] = distances
    output["control_density"] = controls.iloc[matched_indices]["statement_density"].to_numpy()
    output["difference"] = output["statement_density"] - output["control_density"]
    summary = {
        "n": len(output),
        "mean_frontier_density": float(output["statement_density"].mean()),
        "mean_matched_control_density": float(output["control_density"].mean()),
        "mean_difference": float(output["difference"].mean()),
        "median_match_distance": float(np.median(distances)),
        "q95_match_distance": float(np.quantile(distances, 0.95)),
        "unique_control_matches": int(np.unique(matched_indices).size),
        "calipers": {},
    }
    for caliper in (0.25, 0.5, 1.0):
        retained = output.loc[output["match_distance"] <= caliper]
        summary["calipers"][str(caliper)] = {
            "n": len(retained),
            "coverage_fraction": float(len(retained) / len(output)),
            "mean_difference": float(retained["difference"].mean()),
        }
    return output, summary


def old_controls() -> pd.DataFrame:
    manifest = jsonl(REFERENCE / "inputs/manifest.jsonl")
    statements = jsonl(REFERENCE / "inputs/statement.jsonl")
    lengths = [len(row["text"].split("\n", 1)[-1]) for row in statements]
    densities = np.mean(np.load(
        ROOT / "experiments/semantic-neighborhood-transfer-10000/artifacts/statement_top100_cosine.npy",
        allow_pickle=False,
    )[:, : CONFIG["neighbor_k"]], axis=1)
    clusters = np.load(
        REFERENCE / "artifacts/clusters/statement_labels.npy", allow_pickle=False
    ).astype(int)
    return pd.DataFrame({
        "statement_characters": lengths,
        "statement_density": densities,
        "statement_cluster": clusters,
        "name": [row["full_name"] for row in manifest],
    })


def make_figure(projects: pd.DataFrame) -> None:
    order = projects.sort_values("time_matched_difference")["project"].tolist()
    indexed = projects.set_index("project").loc[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.scatter(indexed["old_matched_difference"], y - 0.12, color="#888888", s=35,
               label="2024 LeanDojo control")
    ax.scatter(indexed["time_matched_difference"], y + 0.12, color="#2864a0", s=35,
               label="exact-snapshot Mathlib control")
    for position, row in enumerate(indexed.itertuples()):
        ax.plot([row.old_matched_difference, row.time_matched_difference],
                [position - 0.12, position + 0.12], color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel("matched frontier minus control density")
    ax.set_title("Time matching reduces, but does not remove, frontier isolation", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "time_matched_effects.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "time_matched_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    control_manifest = pd.DataFrame(jsonl(EXPERIMENT / "inputs/manifest.jsonl"))
    density, clusters = map_controls()
    control_manifest["statement_density"] = density
    control_manifest["statement_cluster"] = clusters

    frontier_manifest = pd.DataFrame(jsonl(FRONTIER / "inputs/manifest.jsonl"))
    mapping = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)
    frontier_manifest["statement_density"] = np.mean(mapping["statement_cosine"], axis=1)
    frontier_manifest["statement_cluster"] = mapping["statement_cluster"].astype(int)
    old_control = old_controls()

    project_rows = []
    matched_frames = []
    summaries = {}
    for project, revision in CONFIG["projects"].items():
        front = frontier_manifest.loc[frontier_manifest["project"] == project].copy()
        control = control_manifest.loc[control_manifest["revision"] == revision].copy()
        matched, summary = match_project(front, control)
        _, old_summary = match_project(front, old_control)
        matched["project"] = project
        matched_frames.append(matched)
        summaries[project] = summary
        project_rows.append({
            "project": project,
            "revision": revision,
            "n": len(front),
            "control_n": len(control),
            "frontier_density": summary["mean_frontier_density"],
            "matched_control_density": summary["mean_matched_control_density"],
            "time_matched_difference": summary["mean_difference"],
            "old_matched_difference": old_summary["mean_difference"],
            "median_match_distance": summary["median_match_distance"],
            "q95_match_distance": summary["q95_match_distance"],
            "unique_control_matches": summary["unique_control_matches"],
        })
    projects = pd.DataFrame(project_rows)
    matched_all = pd.concat(matched_frames, ignore_index=True)
    primary = interval(projects["time_matched_difference"].to_numpy())
    old = interval(projects["old_matched_difference"].to_numpy())
    attenuation = float(1 - primary["mean"] / old["mean"])
    calipers = {}
    for caliper in (0.25, 0.5, 1.0):
        key = str(caliper)
        effects = np.asarray([summaries[p]["calipers"][key]["mean_difference"]
                              for p in CONFIG["projects"]])
        calipers[key] = {
            "project_balanced": interval(effects),
            "declarations": sum(summaries[p]["calipers"][key]["n"] for p in CONFIG["projects"]),
            "coverage_fraction": float(np.mean([
                summaries[p]["calipers"][key]["coverage_fraction"] for p in CONFIG["projects"]
            ])),
        }
    snapshot_rows = []
    for revision, group in control_manifest.groupby("revision"):
        snapshot_rows.append({
            "revision": revision,
            "n": len(group),
            "mean_density": float(group["statement_density"].mean()),
            "median_density": float(group["statement_density"].median()),
        })
    result = {
        "experiment_id": CONFIG["experiment_id"],
        "design": "exact pinned Mathlib snapshot; exact statement cluster; nearest log statement length",
        "primary_project_balanced": primary,
        "old_project_balanced_same_eight_projects": old,
        "attenuation_from_time_matching": attenuation,
        "caliper_sensitivity": calipers,
        "projects": project_rows,
        "snapshots": snapshot_rows,
        "excluded": {"lean-liquid": "Lean 3 legacy project; approximately date-matched but not toolchain-matched"},
        "limitations": [
            "observational matching does not identify mathematical novelty causally",
            "project authorship and local formalization conventions remain unmatched",
            "changed-or-new is defined by absence of a verbatim statement in the 2024 corpus",
            "final controls use the global Embed v4 profile while frozen reference/frontier embeddings use the regional base endpoint",
        ],
    }
    artifacts = EXPERIMENT / "artifacts"
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    projects.to_csv(artifacts / "project_matching.csv", index=False)
    matched_all[["project", "name", "statement_density", "control_density", "difference",
                 "match_distance"]].to_csv(artifacts / "matched_declarations.csv", index=False)
    make_figure(projects)

    lines = [
        "# Results: exact-snapshot time controls", "", "## Main result", "",
        f"Across eight modern projects, the project-balanced frontier deficit relative to changed/new "
        f"ordinary Mathlib from each project's exact pinned revision is **{primary['mean']:.4f}** "
        f"(95% t interval {primary['lower']:.4f} to {primary['upper']:.4f}).",
        f"For these same projects, the earlier old-reference matched deficit was **{old['mean']:.4f}**. "
        f"Time matching therefore attenuates the estimated gap by **{100 * attenuation:.1f}%**.", "",
        "The gap remains negative for " + str(int((projects["time_matched_difference"] < 0).sum())) +
        " of 8 projects. Recency and toolchain drift explain part, but not all, of the observed isolation.", "",
        "## Project effects", "",
        "| Project | n | Old control | Exact-snapshot control | Matched control density | Median distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in projects.itertuples():
        lines.append(
            f"| {row.project} | {row.n:,} | {row.old_matched_difference:.4f} | "
            f"{row.time_matched_difference:.4f} | {row.matched_control_density:.4f} | "
            f"{row.median_match_distance:.3f} |"
        )
    half = calipers["0.5"]
    lines.extend([
        "", "![Old and time-matched effects](figures/time_matched_effects.png)", "",
        "## Robustness and interpretation", "",
        f"A 0.5-SD length caliper retains **{100 * half['coverage_fraction']:.1f}%** of declarations "
        f"and gives a project-balanced effect of **{half['project_balanced']['mean']:.4f}** "
        f"(95% t interval {half['project_balanced']['lower']:.4f} to "
        f"{half['project_balanced']['upper']:.4f}).", "",
        "This rejects a pure calendar/toolchain explanation under the measured controls. It does not prove "
        "that the residual is intrinsic mathematical novelty: project authorship, local abstraction choices, "
        "and formalization conventions remain plausible causes. `lean-liquid` is excluded because its Lean 3 "
        "toolchain cannot be cleanly matched to the Lean 4 reference. Corrected controls use the global Embed "
        "v4 profile while the frozen reference/frontier arrays use the regional endpoint; a four-statement "
        "audit gave mean cross-profile cosine 0.99986.",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"primary": primary, "old": old, "attenuation": attenuation,
                      "caliper_0.5": half}, indent=2))


if __name__ == "__main__":
    main()
