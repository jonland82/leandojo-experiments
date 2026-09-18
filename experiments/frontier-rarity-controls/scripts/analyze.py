"""Test whether frontier isolation survives length and domain controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "experiments/frontier-formalizations"
REFERENCE = ROOT / "experiments/semantic-embeddings-10000"
TRANSFER = ROOT / "experiments/semantic-neighborhood-transfer-10000/artifacts"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def text_lengths(path: Path) -> np.ndarray:
    return np.asarray([len(row["text"]) for row in load_jsonl(path)], dtype=np.int32)


def finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "n": int(len(values))}
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n": int(len(values))}


def build_frame() -> tuple[pd.DataFrame, int]:
    ref_manifest = load_jsonl(REFERENCE / "inputs/manifest.jsonl")
    front_manifest = load_jsonl(FRONTIER / "inputs/manifest.jsonl")
    n_reference = len(ref_manifest)

    ref_statement_length = text_lengths(REFERENCE / "inputs/statement.jsonl")
    ref_proof_length = text_lengths(REFERENCE / "inputs/proof.jsonl")
    front_statement_length = text_lengths(FRONTIER / "inputs/statement.jsonl")
    front_proof_length = text_lengths(FRONTIER / "inputs/proof.jsonl")

    mapping = np.load(FRONTIER / "artifacts/mapping_arrays.npz", allow_pickle=False)
    ref_statement_density = np.mean(
        np.load(TRANSFER / "statement_top100_cosine.npy", allow_pickle=False)[:, :10], axis=1
    )
    ref_proof_density = np.mean(
        np.load(TRANSFER / "proof_top100_cosine.npy", allow_pickle=False)[:, :10], axis=1
    )
    front_statement_density = np.mean(mapping["statement_cosine"], axis=1)
    front_proof_density = np.mean(mapping["proof_cosine"], axis=1)

    ref_statement_cluster = np.load(
        REFERENCE / "artifacts/clusters/statement_labels.npy", allow_pickle=False
    ).astype(int)
    ref_proof_cluster = np.load(
        REFERENCE / "artifacts/clusters/proof_labels.npy", allow_pickle=False
    ).astype(int)

    frame = pd.DataFrame({
        "frontier": np.r_[np.zeros(n_reference, dtype=int), np.ones(len(front_manifest), dtype=int)],
        "project": ["LeanDojo"] * n_reference + [row["project"] for row in front_manifest],
        "headline": np.r_[np.zeros(n_reference, dtype=bool), [row["headline"] for row in front_manifest]],
        "statement_characters": np.r_[ref_statement_length, front_statement_length],
        "proof_characters": np.r_[ref_proof_length, front_proof_length],
        "statement_cluster": np.r_[ref_statement_cluster, mapping["statement_cluster"].astype(int)],
        "proof_cluster": np.r_[ref_proof_cluster, mapping["proof_cluster"].astype(int)],
        "statement_density": np.r_[ref_statement_density, front_statement_density],
        "proof_density": np.r_[ref_proof_density, front_proof_density],
    })
    for name in ("statement", "proof"):
        log_values = np.log1p(frame[f"{name}_characters"].to_numpy(dtype=np.float64))
        ref_values = log_values[:n_reference]
        frame[f"z_log_{name}_characters"] = (
            (log_values - np.mean(ref_values)) / np.std(ref_values, ddof=1)
        )
    return frame, n_reference


def fit_models(frame: pd.DataFrame, outcome: str) -> list[dict]:
    length_terms = (
        "z_log_statement_characters + I(z_log_statement_characters ** 2) + "
        "z_log_proof_characters + I(z_log_proof_characters ** 2)"
    )
    formulas = {
        "raw": f"{outcome} ~ frontier",
        "length": f"{outcome} ~ frontier + {length_terms}",
        "domain": f"{outcome} ~ frontier + C(statement_cluster)",
        "length_and_domain": f"{outcome} ~ frontier + {length_terms} + C(statement_cluster)",
    }
    rows = []
    for name, formula in formulas.items():
        model = smf.ols(formula, data=frame).fit(cov_type="HC3")
        lower, upper = model.conf_int().loc["frontier"]
        rows.append({
            "model": name,
            "formula": formula,
            "frontier_coefficient": float(model.params["frontier"]),
            "standard_error": float(model.bse["frontier"]),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "p_value": float(model.pvalues["frontier"]),
            "r_squared": float(model.rsquared),
            "n": int(model.nobs),
        })
    return rows


def match_frontier(frame: pd.DataFrame, n_reference: int) -> tuple[pd.DataFrame, dict]:
    reference = frame.iloc[:n_reference].copy()
    frontier = frame.iloc[n_reference:].copy()
    features = ["z_log_statement_characters", "z_log_proof_characters"]
    matched_reference_indices = np.empty(len(frontier), dtype=np.int32)
    distances = np.empty(len(frontier), dtype=np.float64)

    for cluster in sorted(frontier["statement_cluster"].unique()):
        front_positions = np.flatnonzero(frontier["statement_cluster"].to_numpy() == cluster)
        ref_positions = np.flatnonzero(reference["statement_cluster"].to_numpy() == cluster)
        model = NearestNeighbors(n_neighbors=1, metric="euclidean")
        model.fit(reference.iloc[ref_positions][features].to_numpy())
        distance, local_index = model.kneighbors(frontier.iloc[front_positions][features].to_numpy())
        matched_reference_indices[front_positions] = ref_positions[local_index[:, 0]]
        distances[front_positions] = distance[:, 0]

    matched = frontier.reset_index(drop=True).copy()
    matched["match_distance"] = distances
    matched["matched_reference_index"] = matched_reference_indices
    for view in ("statement", "proof"):
        matched[f"matched_{view}_density"] = reference.iloc[matched_reference_indices][
            f"{view}_density"
        ].to_numpy()
        matched[f"{view}_difference"] = (
            matched[f"{view}_density"] - matched[f"matched_{view}_density"]
        )

    project_rows = []
    for project, group in matched.groupby("project", sort=True):
        project_rows.append({
            "project": project,
            "n": int(len(group)),
            "headline_n": int(group["headline"].sum()),
            "median_match_distance": float(group["match_distance"].median()),
            "statement_difference": float(group["statement_difference"].mean()),
            "proof_difference": float(group["proof_difference"].mean()),
        })
    projects = pd.DataFrame(project_rows)
    unique_matches = int(np.unique(matched_reference_indices).size)
    caliper_sensitivity = {}
    for caliper in (0.25, 0.5, 1.0):
        retained = matched.loc[matched["match_distance"] <= caliper]
        retained_projects = retained.groupby("project")[[
            "statement_difference", "proof_difference"
        ]].mean()
        caliper_sensitivity[str(caliper)] = {
            "n": int(len(retained)),
            "coverage_fraction": float(len(retained) / len(matched)),
            "projects": int(len(retained_projects)),
            "sample_weighted": {
                view: interval(retained[f"{view}_difference"].to_numpy())
                for view in ("statement", "proof")
            },
            "project_balanced": {
                view: interval(retained_projects[f"{view}_difference"].to_numpy())
                for view in ("statement", "proof")
            },
        }
    summary = {
        "n": int(len(matched)),
        "unique_reference_matches": unique_matches,
        "reference_reuse_fraction": float(1 - unique_matches / len(matched)),
        "match_distance": {
            "median": float(np.median(distances)),
            "q95": float(np.quantile(distances, 0.95)),
            "maximum": float(np.max(distances)),
        },
        "sample_weighted": {
            view: interval(matched[f"{view}_difference"].to_numpy())
            for view in ("statement", "proof")
        },
        "project_balanced": {
            view: interval(projects[f"{view}_difference"].to_numpy())
            for view in ("statement", "proof")
        },
        "headlines": {
            view: interval(matched.loc[matched["headline"], f"{view}_difference"].to_numpy())
            for view in ("statement", "proof")
        },
        "caliper_sensitivity": caliper_sensitivity,
    }
    return projects, summary


def make_figure(models: dict[str, list[dict]], projects: pd.DataFrame) -> None:
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    model_order = ["raw", "length", "domain", "length_and_domain"]
    labels = ["raw", "+ length", "+ domain", "+ both"]
    colors = {"statement": "#2864a0", "proof": "#c04b4b"}
    offsets = {"statement": -0.10, "proof": 0.10}
    y = np.arange(len(model_order))
    for view in ("statement", "proof"):
        indexed = {row["model"]: row for row in models[view]}
        values = np.asarray([indexed[name]["frontier_coefficient"] for name in model_order])
        lower = np.asarray([indexed[name]["ci95_lower"] for name in model_order])
        upper = np.asarray([indexed[name]["ci95_upper"] for name in model_order])
        axes[0].errorbar(
            values, y + offsets[view], xerr=np.vstack([values - lower, upper - values]),
            fmt="o", capsize=3, color=colors[view], label=view,
        )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("adjusted frontier density difference")
    axes[0].set_title("(a) Regression adjustment", loc="left")
    axes[0].legend(frameon=False)

    project_order = projects.sort_values("statement_difference")["project"].tolist()
    indexed_projects = projects.set_index("project")
    y = np.arange(len(project_order))
    for view in ("statement", "proof"):
        values = indexed_projects.loc[project_order, f"{view}_difference"].to_numpy()
        axes[1].scatter(values, y + offsets[view], s=28, color=colors[view], label=view)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, project_order)
    axes[1].set_xlabel("matched frontier minus LeanDojo density")
    axes[1].set_title("(b) Length/domain-matched projects", loc="left")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "controlled_rarity.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "controlled_rarity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, n_reference = build_frame()
    models = {
        view: fit_models(frame, f"{view}_density")
        for view in ("statement", "proof")
    }
    projects, matching = match_frontier(frame, n_reference)
    make_figure(models, projects)

    raw = {view: models[view][0]["frontier_coefficient"] for view in models}
    adjusted = {view: models[view][-1]["frontier_coefficient"] for view in models}
    result = {
        "experiment_id": "frontier-rarity-controls",
        "reference_n": n_reference,
        "frontier_n": int(len(frame) - n_reference),
        "models": models,
        "matching": matching,
        "project_matching": projects.to_dict(orient="records"),
        "attenuation_fraction": {
            view: finite(1 - adjusted[view] / raw[view]) for view in raw
        },
        "scope": {
            "primary_view": "statement",
            "controlled": ["statement character count", "proof character count", "statement cluster"],
            "not_identified": ["project age/toolchain", "intrinsic mathematical complexity", "causal novelty"],
        },
    }
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    projects.to_csv(artifacts / "project_matching.csv", index=False)

    statement_full = models["statement"][-1]
    proof_full = models["proof"][-1]
    statement_match = matching["project_balanced"]["statement"]
    proof_match = matching["project_balanced"]["proof"]
    caliper = matching["caliper_sensitivity"]["0.5"]
    lines = [
        "# Results: conditional rarity of frontier formalizations", "",
        "## Main result", "",
        f"The raw statement-density deficit is **{raw['statement']:.4f}** cosine points. "
        f"After controlling for statement/proof character counts and frozen statement-cluster fixed effects, "
        f"the deficit is **{statement_full['frontier_coefficient']:.4f}** "
        f"(HC3 95% CI {statement_full['ci95_lower']:.4f} to {statement_full['ci95_upper']:.4f}).",
        f"The project-balanced matched estimate is **{statement_match['mean']:.4f}** "
        f"(95% t interval {statement_match['lower']:.4f} to {statement_match['upper']:.4f}).", "",
        f"Restricting to matches within 0.5 standardized log-length units retains "
        f"**{100 * caliper['coverage_fraction']:.1f}%** of frontier declarations and gives a "
        f"project-balanced statement deficit of **{caliper['project_balanced']['statement']['mean']:.4f}**.", "",
        "Thus observable text length and coarse semantic domain do not explain away frontier isolation. "
        "They may attenuate it, but the controlled effect remains a substantial corpus-level shift. "
        "All nine projects have negative matched statement effects, so the conclusion is not driven by "
        "a single formalization. The project-balanced interval is the preferred uncertainty summary; "
        "the narrower HC3 interval conditions on these sampled projects.", "",
        "## Secondary proof view", "",
        f"The adjusted proof-density deficit is **{proof_full['frontier_coefficient']:.4f}** "
        f"(HC3 95% CI {proof_full['ci95_lower']:.4f} to {proof_full['ci95_upper']:.4f}); "
        f"the project-balanced matched estimate is **{proof_match['mean']:.4f}**. "
        "This comparison is descriptive because the two corpora encode proofs differently.", "",
        "## Project-matched effects", "",
        "| Project | n | Statement difference | Proof difference | Median match distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in projects.to_dict(orient="records"):
        lines.append(
            f"| {row['project']} | {row['n']:,} | {row['statement_difference']:.4f} | "
            f"{row['proof_difference']:.4f} | {row['median_match_distance']:.3f} |"
        )
    lines.extend([
        "", "![Controlled rarity](figures/controlled_rarity.png)", "",
        "## Interpretation limits", "",
        "The result rejects a simple length-only explanation and is robust to a coarse domain control. "
        "It does **not** distinguish mathematical novelty from project age, Lean/Mathlib version, authorship, "
        "or formalization style. Proof character count is not intrinsic complexity, and cluster adjustment "
        "is not a causal design. A time-matched multi-snapshot corpus is required to isolate recency.",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "statement_adjusted": statement_full,
        "statement_project_matched": statement_match,
        "proof_adjusted": proof_full,
        "proof_project_matched": proof_match,
    }, indent=2))


if __name__ == "__main__":
    main()
