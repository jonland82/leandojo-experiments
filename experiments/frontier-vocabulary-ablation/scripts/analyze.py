"""Measure frontier isolation after project-global vocabulary anonymization."""

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


def density(queries: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    output = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        similarity = queries[start:start + 512] @ reference.T
        output[start:start + 512] = np.mean(np.partition(similarity, -k, axis=1)[:, -k:], axis=1)
    return output


def interval(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": len(values)}


def main() -> None:
    style_manifest = pd.DataFrame(jsonl(STYLE / "inputs/manifest.jsonl"))
    style_inputs = jsonl(STYLE / "inputs/alpha_normalized.jsonl")
    style_embeddings = normalize(np.load(
        STYLE / "artifacts/embeddings/alpha_normalized.npy", allow_pickle=False))
    reference_mask = style_manifest["cohort"].to_numpy() == "reference"
    control_mask = style_manifest["cohort"].to_numpy() == "control"
    frontier_mask = style_manifest["cohort"].to_numpy() == "frontier"
    reference_vectors = style_embeddings[reference_mask]
    control_vectors = style_embeddings[control_mask]
    frontier_vectors = normalize(np.load(
        EXPERIMENT / "artifacts/embeddings/vocabulary_anonymized.npy", allow_pickle=False))
    control_density = density(control_vectors, reference_vectors, CONFIG["neighbor_k"])
    frontier_density = density(frontier_vectors, reference_vectors, CONFIG["neighbor_k"])
    centers = normalize(np.load(REFERENCE / "artifacts/clusters/statement_centers.npy", allow_pickle=False))
    raw_controls = normalize(np.load(TIME / "artifacts/embeddings/statement.npy", allow_pickle=False))
    control_clusters = np.argmax(raw_controls @ centers.T, axis=1).astype(np.int16)
    frontier_clusters = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)[
        "statement_cluster"].astype(np.int16)
    lengths = np.asarray([row["statement_characters"] for row in style_inputs])
    anonymized_inputs = jsonl(EXPERIMENT / "inputs/vocabulary_anonymized.jsonl")
    controls = style_manifest.loc[control_mask, ["revision"]].reset_index(drop=True)
    controls["length"] = lengths[control_mask]
    controls["density"] = control_density
    controls["cluster"] = control_clusters
    frontier = style_manifest.loc[frontier_mask, ["project"]].reset_index(drop=True)
    frontier["length"] = [row["statement_characters"] for row in anonymized_inputs]
    frontier["density"] = frontier_density
    frontier["cluster"] = frontier_clusters
    effects = []
    for project, revision in TIME_CONFIG["projects"].items():
        front = frontier[frontier["project"] == project].copy()
        control = controls[controls["revision"] == revision].copy()
        matches = np.empty(len(front), dtype=np.int32)
        for cluster in sorted(front["cluster"].unique()):
            fp = np.flatnonzero(front["cluster"].to_numpy() == cluster)
            cp = np.flatnonzero(control["cluster"].to_numpy() == cluster)
            model = NearestNeighbors(n_neighbors=1).fit(np.log1p(control.iloc[cp]["length"]).to_numpy()[:, None])
            _, local = model.kneighbors(np.log1p(front.iloc[fp]["length"]).to_numpy()[:, None])
            matches[fp] = cp[local[:, 0]]
        effect = float(np.mean(front["density"].to_numpy() - control.iloc[matches]["density"].to_numpy()))
        effects.append({"project": project, "n": len(front), "effect": effect})
    table = pd.DataFrame(effects)
    result_interval = interval(table["effect"].to_numpy())
    baseline = json.loads((STYLE / "artifacts/analysis.json").read_text(encoding="utf-8"))[
        "representations"]["alpha_normalized"]
    raw_baseline = json.loads((STYLE / "artifacts/analysis.json").read_text(encoding="utf-8"))[
        "representations"]["raw"]
    attenuation = float(1 - result_interval["mean"] / baseline["mean"])
    combined_attenuation = float(1 - result_interval["mean"] / raw_baseline["mean"])
    result = {"experiment_id": CONFIG["experiment_id"], "project_balanced_effect": result_interval,
              "alpha_normalized_baseline": baseline, "raw_baseline": raw_baseline,
              "attenuation_from_alpha_baseline": attenuation,
              "combined_attenuation_from_raw_baseline": combined_attenuation,
              "projects": effects,
              "interpretation": "Attenuation estimates sensitivity to conservatively resolved project-global names; unresolved vocabulary and mathematical structure remain.",
              "limitations": ["symbol resolution is lexical rather than elaborated",
                              "ambiguous unqualified project symbols are deliberately retained",
                              "placeholder anonymization removes symbol identity semantics as well as spelling"]}
    artifacts = EXPERIMENT / "artifacts"
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table.to_csv(artifacts / "project_effects.csv", index=False)
    joined = table.merge(pd.DataFrame(json.loads((STYLE / "artifacts/analysis.json").read_text(
        encoding="utf-8"))["variants"]["alpha_normalized"]["projects"])[["project", "effect"]],
        on="project", suffixes=("_vocabulary", "_alpha"))
    order = joined.sort_values("effect_vocabulary")["project"]
    plot = joined.set_index("project").loc[order]
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(plot["effect_alpha"], y - 0.12, label="bound names removed", color="#777777")
    ax.scatter(plot["effect_vocabulary"], y + 0.12, label="+ project globals removed", color="#1d6996")
    for position, row in enumerate(plot.itertuples()):
        ax.plot([row.effect_alpha, row.effect_vocabulary], [position - 0.12, position + 0.12],
                color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [value.replace("navier-stokes-euler", "Navier–Stokes") for value in order])
    ax.set_xlabel("frontier − exact-snapshot control density")
    ax.set_title("Project-vocabulary ablation", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "vocabulary_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "vocabulary_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    lines = ["# Results: project-vocabulary ablation", "", "## Main result", "",
        f"The alpha-normalized project-balanced density deficit changes from **{baseline['mean']:.4f}** "
        f"to **{result_interval['mean']:.4f}** (95% project-level t interval "
        f"{result_interval['lower']:.4f} to {result_interval['upper']:.4f}) after conservatively anonymizing "
        f"project-defined global identifiers. This is **{100 * attenuation:.1f}% attenuation** beyond bound-name "
        f"normalization and **{100 * combined_attenuation:.1f}% total attenuation** from the raw baseline.", "",
        "## Project effects", "", "| Project | n | Vocabulary-anonymized effect |", "|---|---:|---:|"]
    for row in table.itertuples():
        lines.append(f"| {row.project} | {row.n} | {row.effect:.4f} |")
    lines.extend(["", "![Vocabulary ablation](figures/vocabulary_ablation.png)", "",
        "## Interpretation", "", "A large attenuation would support repository dialect as the primary mechanism; "
        "a persistent negative gap would show that global-name spelling alone is insufficient. Resolution is "
        "conservative, so this test can underestimate the full vocabulary contribution."])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"effect": result_interval, "baseline": baseline, "attenuation": attenuation,
                      "combined_attenuation": combined_attenuation}, indent=2))


if __name__ == "__main__":
    main()
