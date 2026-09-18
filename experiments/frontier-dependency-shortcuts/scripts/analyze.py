"""Compare dependency-cone compression of headline and matched declarations."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "n": len(values)}
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def cone_metrics(root: int, nodes: list[dict], dependents: list[set[int]]) -> dict:
    distance = {root: 0}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for dependency in nodes[current]["dependencies"]:
            if dependency not in distance:
                distance[dependency] = distance[current] + 1
                queue.append(dependency)
    cone = set(distance) - {root}
    savings = []
    for index in cone:
        reuse = len(dependents[index])
        body = nodes[index]["proof_characters"]
        reference = len(nodes[index]["full_name"])
        savings.append(max(0, (reuse - 1) * body - reuse * reference))
    total_cost = float(sum(nodes[index]["proof_characters"] for index in cone))
    return {
        "direct_dependencies": len(nodes[root]["dependencies"]),
        "closure_size": len(cone),
        "closure_depth": max(distance.values(), default=0),
        "cone_proof_characters": int(total_cost),
        "reusable_fraction": float(np.mean([len(dependents[index]) >= 2 for index in cone])) if cone else 0.0,
        "mean_log_reuse": float(np.mean([np.log1p(len(dependents[index])) for index in cone])) if cone else 0.0,
        "compression_savings": float(sum(savings)),
        "compression_ratio": float(sum(savings) / max(1.0, total_cost)),
        "compression_fraction": float(sum(savings) / max(1.0, sum(savings) + total_cost)),
    }


def ancestor_set(root: int, nodes: list[dict]) -> set[int]:
    seen = {root}
    stack = [root]
    while stack:
        for dependency in nodes[stack.pop()]["dependencies"]:
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)
    return seen - {root}


def matched_controls(headline: dict, candidates: list[dict], count: int) -> list[int]:
    same_kind = [row for row in candidates if row["kind"] == headline["kind"]]
    pool = same_kind if len(same_kind) >= count else candidates
    columns = np.asarray([[np.log1p(row["proof_characters"]), row["relative_position"],
                           np.log1p(row["file_declarations"]),
                           np.log1p(row["statement_characters"])] for row in pool])
    query = np.asarray([[np.log1p(headline["proof_characters"]), headline["relative_position"],
                         np.log1p(headline["file_declarations"]),
                         np.log1p(headline["statement_characters"])]])
    scaler = StandardScaler().fit(columns)
    distances = np.linalg.norm(scaler.transform(columns) - scaler.transform(query), axis=1)
    order = np.argsort(distances, kind="stable")[:min(count, len(pool))]
    return [pool[index]["id"] for index in order]


def make_figure(projects: pd.DataFrame) -> None:
    order = projects.sort_values("compression_fraction_lift")["project"].tolist()
    table = projects.set_index("project").loc[order]
    labels = [value.replace("navier-stokes-euler", "Navier–Stokes") for value in order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8), sharey=True)
    axes[0].barh(y, table["compression_fraction_lift"], color="#1d6996")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("headline − matched compression fraction")
    axes[0].set_yticks(y, labels)
    axes[0].set_title("Reusable compression", loc="left")
    axes[1].barh(y, table["log_closure_lift"], color="#edae49")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("headline − matched log closure")
    axes[1].set_title("Prerequisite reach", loc="left")
    fig.suptitle("Headline dependency cones versus within-project controls", x=0.08, ha="left")
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "dependency_shortcuts.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "dependency_shortcuts.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    nodes = jsonl(EXPERIMENT / "artifacts/graph_nodes.jsonl")
    dependents = [set() for _ in nodes]
    for node in nodes:
        for dependency in node["dependencies"]:
            dependents[dependency].add(node["id"])
    by_project: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        by_project[node["project"]].append(node)
    metric_cache: dict[int, dict] = {}

    def metrics(index: int) -> dict:
        if index not in metric_cache:
            metric_cache[index] = cone_metrics(index, nodes, dependents)
        return metric_cache[index]

    comparisons = []
    for project, project_nodes in by_project.items():
        headlines = [row for row in project_nodes if row["headline"]]
        frontier_cone = set().union(*(ancestor_set(row["id"], nodes) for row in headlines))
        candidates = [row for row in project_nodes if not row["headline"]
                      and row["id"] not in frontier_cone
                      and row["kind"] in {"theorem", "lemma"} and row["proof_characters"] > 0]
        for headline in headlines:
            control_ids = matched_controls(headline, candidates, CONFIG["matched_controls_per_headline"])
            headline_metrics = metrics(headline["id"])
            control_metrics = [metrics(index) for index in control_ids]
            row = {"project": project, "headline": headline["headline_name"],
                   "file_path": headline["file_path"], "line": headline["line"],
                   "proof_characters": headline["proof_characters"], "matched_controls": len(control_ids)}
            for key, value in headline_metrics.items():
                row[f"headline_{key}"] = value
                row[f"control_{key}"] = float(np.mean([item[key] for item in control_metrics]))
                if key in {"closure_size", "cone_proof_characters", "compression_savings"}:
                    row[f"{key}_lift"] = float(np.log1p(value) - np.mean(
                        [np.log1p(item[key]) for item in control_metrics]))
                else:
                    row[f"{key}_lift"] = float(value - np.mean([item[key] for item in control_metrics]))
            comparisons.append(row)
    table = pd.DataFrame(comparisons)
    table["log_closure_lift"] = table["closure_size_lift"]
    project_rows = []
    for project, group in table.groupby("project"):
        project_rows.append({"project": project, "headlines": len(group),
            "compression_fraction_lift": float(group["compression_fraction_lift"].mean()),
            "log_closure_lift": float(group["log_closure_lift"].mean()),
            "reusable_fraction_lift": float(group["reusable_fraction_lift"].mean()),
            "closure_depth_lift": float(group["closure_depth_lift"].mean())})
    projects = pd.DataFrame(project_rows)
    primary = projects[projects["project"].isin(CONFIG["primary_projects"])]
    keys = ["compression_fraction_lift", "log_closure_lift", "reusable_fraction_lift", "closure_depth_lift"]
    aggregates = {key: interval(primary[key].to_numpy()) for key in keys}
    visible_source = primary[primary["project"] != "flt"]
    sensitivity = {key: interval(visible_source[key].to_numpy()) for key in keys}
    headline_level = {key: interval(table.loc[
        table["project"].isin(CONFIG["primary_projects"]), key].to_numpy()) for key in keys}
    result = {"experiment_id": CONFIG["experiment_id"],
        "graph": json.loads((EXPERIMENT / "artifacts/preparation.json").read_text(encoding="utf-8")),
        "primary_project_balanced": aggregates,
        "sensitivity_excluding_macro_obscured_flt": sensitivity,
        "headline_level_descriptive": headline_level,
        "projects": project_rows,
        "interpretation": "Positive lift means a named result has more reusable compression or prerequisite reach than source-architecture-matched declarations in the same project.",
        "limitations": [
            "lexical references undercount elaborated, notation-hidden, generated, and typeclass dependencies",
            "compression savings use source characters as a minimum-description-length proxy",
            "within-project matching is descriptive and does not identify a causal novelty effect",
            "Lean 3 lean-liquid is reported descriptively but excluded from the primary eight-project interval"]}
    artifacts = EXPERIMENT / "artifacts"
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table.to_csv(artifacts / "headline_comparisons.csv", index=False)
    projects.to_csv(artifacts / "project_effects.csv", index=False)
    make_figure(projects)
    compression, closure, reusable = (aggregates[key] for key in keys[:3])
    lines = ["# Results: dependency shortcuts and compression", "", "## Main result", "",
        f"Across the eight modern projects, named headline results have a project-balanced "
        f"compression-fraction lift of **{compression['mean']:.3f}** (95% t interval "
        f"{compression['lower']:.3f} to {compression['upper']:.3f}) relative to declarations "
        "matched within project on kind, proof length, source position, file size, and statement length.", "",
        f"Their prerequisite cones have a mean log-closure lift of **{closure['mean']:.3f}** "
        f"({closure['lower']:.3f} to {closure['upper']:.3f}). The reusable-lemma fraction changes by "
        f"**{reusable['mean']:.3f}** ({reusable['lower']:.3f} to {reusable['upper']:.3f}).", "",
        f"Excluding FLT, whose `p2m_exact_reverting` macro hides its internal dependency chain from this "
        f"source-static parser, the compression lift is **{sensitivity['compression_fraction_lift']['mean']:.3f}** "
        f"({sensitivity['compression_fraction_lift']['lower']:.3f} to "
        f"{sensitivity['compression_fraction_lift']['upper']:.3f}). In that sensitivity set, headline cones "
        f"are **{sensitivity['closure_depth_lift']['mean']:.2f} dependency steps deeper** "
        f"({sensitivity['closure_depth_lift']['lower']:.2f} to "
        f"{sensitivity['closure_depth_lift']['upper']:.2f}).", "",
        "## Project effects", "", "| Project | Headlines | Compression lift | Log-closure lift | Reusable-fraction lift |",
        "|---|---:|---:|---:|---:|"]
    for row in projects.itertuples():
        lines.append(f"| {row.project} | {row.headlines} | {row.compression_fraction_lift:.3f} | "
                     f"{row.log_closure_lift:.3f} | {row.reusable_fraction_lift:.3f} |")
    lines.extend(["", "![Dependency shortcut effects](figures/dependency_shortcuts.png)", "",
        "Positive values indicate that the named theorem reaches a larger or more reusable internal library "
        "than comparable declarations. Compression lift is the closest tested operational analogue of a "
        "useful mathematical shortcut.", "", "## Interpretation", "",
        "This test distinguishes mere length from architectural leverage: controls are matched on surface "
        "size and location, excluded from every headline cone, and evaluated over transitive prerequisites. "
        "The current evidence favors deeper integration over exceptional reusable compression. It remains "
        "an approximate source-graph result, not an elaborator-certified dependency or a causal measure of "
        "mathematical creativity."])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"headlines": len(table), "primary_projects": len(primary),
                      "project_balanced": aggregates}, indent=2))


if __name__ == "__main__":
    main()
