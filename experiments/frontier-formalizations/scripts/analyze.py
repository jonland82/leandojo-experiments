"""Map frontier declarations into the frozen 10,000-item LeanDojo spaces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/semantic-embeddings-10000"
TRANSFER = ROOT / "experiments/semantic-neighborhood-transfer-10000/artifacts"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def nearest(frontier: np.ndarray, reference: np.ndarray, k: int = 10, chunk: int = 256):
    indices = np.empty((len(frontier), k), dtype=np.int32)
    cosine = np.empty((len(frontier), k), dtype=np.float32)
    for start in range(0, len(frontier), chunk):
        end = min(len(frontier), start + chunk)
        scores = frontier[start:end] @ reference.T
        candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
        values = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(values, axis=1)[:, ::-1]
        indices[start:end] = np.take_along_axis(candidates, order, axis=1)
        cosine[start:end] = np.take_along_axis(values, order, axis=1)
        if end % 2048 == 0 or end == len(frontier):
            print(f"neighbors: {end:,}/{len(frontier):,}", flush=True)
    return indices, cosine


def finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def stats(values: np.ndarray) -> dict:
    return {
        "mean": finite(np.mean(values)),
        "median": finite(np.median(values)),
        "q05": finite(np.quantile(values, 0.05)),
        "q95": finite(np.quantile(values, 0.95)),
    }


def main() -> None:
    manifest = load_jsonl(EXPERIMENT / "inputs/manifest.jsonl")
    reference_manifest = load_jsonl(REFERENCE / "inputs/manifest.jsonl")
    projects = sorted({row["project"] for row in manifest})
    project_indices = {
        project: np.array([i for i, row in enumerate(manifest) if row["project"] == project])
        for project in projects
    }
    headline_indices = np.array([i for i, row in enumerate(manifest) if row["headline"]])
    rng = np.random.default_rng(0)

    arrays = {}
    neighbor_indices = {}
    neighbor_cosine = {}
    assignments = {}
    reference_density = {}
    for view in ("statement", "proof"):
        ref = normalize(np.load(REFERENCE / f"artifacts/embeddings/{view}.npy", allow_pickle=False))
        front = normalize(np.load(EXPERIMENT / f"artifacts/embeddings/{view}.npy", allow_pickle=False))
        arrays[view] = (ref, front)
        idx, cos = nearest(front, ref)
        neighbor_indices[view], neighbor_cosine[view] = idx, cos
        centers = normalize(np.load(REFERENCE / f"artifacts/clusters/{view}_centers.npy", allow_pickle=False))
        assignments[view] = np.argmax(front @ centers.T, axis=1).astype(np.int16)
        if view in ("statement", "proof"):
            reference_density[view] = np.mean(
                np.load(TRANSFER / f"{view}_top100_cosine.npy", allow_pickle=False)[:, :10], axis=1
            )

    statement_neighbors = neighbor_indices["statement"]
    proof_neighbors = neighbor_indices["proof"]
    overlap = np.array([
        len(set(a).intersection(b)) / len(set(a).union(b))
        for a, b in zip(statement_neighbors, proof_neighbors)
    ])
    ref_proof, frontier_proof = arrays["proof"]
    proof_on_statement_neighbors = np.mean(
        np.sum(frontier_proof[:, None, :] * ref_proof[statement_neighbors], axis=2), axis=1
    )
    random_indices = rng.integers(0, len(ref_proof), size=statement_neighbors.shape)
    proof_random = np.mean(
        np.sum(frontier_proof[:, None, :] * ref_proof[random_indices], axis=2), axis=1
    )
    transfer_delta = proof_on_statement_neighbors - proof_random

    project_summaries = {}
    statement_cutoff = float(np.quantile(reference_density["statement"], 0.05))
    proof_cutoff = float(np.quantile(reference_density["proof"], 0.05))
    for project, idx in project_indices.items():
        project_summaries[project] = {
            "n": int(len(idx)),
            "headline_n": int(sum(manifest[i]["headline"] for i in idx)),
            "statement_top1": stats(neighbor_cosine["statement"][idx, 0]),
            "statement_top10_mean": stats(np.mean(neighbor_cosine["statement"][idx], axis=1)),
            "proof_top1": stats(neighbor_cosine["proof"][idx, 0]),
            "proof_top10_mean": stats(np.mean(neighbor_cosine["proof"][idx], axis=1)),
            "statement_below_reference_q05_fraction": float(
                np.mean(np.mean(neighbor_cosine["statement"][idx], axis=1) < statement_cutoff)
            ),
            "proof_below_reference_q05_fraction": float(
                np.mean(np.mean(neighbor_cosine["proof"][idx], axis=1) < proof_cutoff)
            ),
            "statement_proof_neighbor_jaccard": stats(overlap[idx]),
            "proof_similarity_on_statement_neighbors": stats(proof_on_statement_neighbors[idx]),
            "random_proof_similarity": stats(proof_random[idx]),
            "transfer_delta": stats(transfer_delta[idx]),
            "statement_clusters": dict(sorted(Counter(map(int, assignments["statement"][idx])).items())),
            "proof_clusters": dict(sorted(Counter(map(int, assignments["proof"][idx])).items())),
        }

    headlines = []
    for i in headline_indices:
        row = manifest[int(i)]
        nearest_rows = []
        for ref_i, similarity in zip(neighbor_indices["statement"][i, :5], neighbor_cosine["statement"][i, :5]):
            ref_row = reference_manifest[int(ref_i)]
            nearest_rows.append({
                "full_name": ref_row["full_name"],
                "file_path": ref_row["file_path"],
                "cosine": float(similarity),
            })
        headlines.append({
            "project": row["project"],
            "name": row["name"],
            "file_path": row["file_path"],
            "line": row["line"],
            "statement_top1": float(neighbor_cosine["statement"][i, 0]),
            "proof_top1": float(neighbor_cosine["proof"][i, 0]),
            "statement_cluster": int(assignments["statement"][i]),
            "proof_cluster": int(assignments["proof"][i]),
            "neighbor_jaccard": float(overlap[i]),
            "transfer_delta": float(transfer_delta[i]),
            "nearest_statement_neighbors": nearest_rows,
        })

    statement_ref_density = reference_density["statement"]
    proof_ref_density = reference_density["proof"]
    result = {
        "experiment_id": "frontier-formalizations",
        "declarations": len(manifest),
        "headlines": len(headlines),
        "reference_declarations": len(reference_manifest),
        "reference_baselines": {
            "statement_top10_mean": stats(statement_ref_density),
            "proof_top10_mean": stats(proof_ref_density),
            "statement_q05": statement_cutoff,
            "proof_q05": proof_cutoff,
        },
        "overall": {
            "statement_top10_mean": stats(np.mean(neighbor_cosine["statement"], axis=1)),
            "proof_top10_mean": stats(np.mean(neighbor_cosine["proof"], axis=1)),
            "statement_below_reference_q05_fraction": float(
                np.mean(np.mean(neighbor_cosine["statement"], axis=1) < statement_cutoff)
            ),
            "proof_below_reference_q05_fraction": float(
                np.mean(np.mean(neighbor_cosine["proof"], axis=1) < proof_cutoff)
            ),
            "statement_proof_neighbor_jaccard": stats(overlap),
            "transfer_delta": stats(transfer_delta),
        },
        "projects": project_summaries,
        "headline_results": headlines,
    }
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        artifacts / "mapping_arrays.npz",
        statement_neighbors=neighbor_indices["statement"],
        statement_cosine=neighbor_cosine["statement"],
        proof_neighbors=neighbor_indices["proof"],
        proof_cosine=neighbor_cosine["proof"],
        statement_cluster=assignments["statement"],
        proof_cluster=assignments["proof"],
        overlap=overlap,
        transfer_delta=transfer_delta,
    )

    ref_statement, front_statement = arrays["statement"]
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    ref_xy = pca.fit_transform(ref_statement)
    front_xy = pca.transform(front_statement)
    figures = EXPERIMENT / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], s=3, alpha=0.12, color="#808080", label="LeanDojo reference")
    palette = plt.cm.tab10(np.linspace(0, 1, len(projects)))
    for color, project in zip(palette, projects):
        idx = project_indices[project]
        shown = idx if len(idx) <= 600 else rng.choice(idx, 600, replace=False)
        ax.scatter(front_xy[shown, 0], front_xy[shown, 1], s=8, alpha=0.45, color=color, label=project)
    ax.scatter(front_xy[headline_indices, 0], front_xy[headline_indices, 1], s=65, marker="*", color="black", label="headline")
    ax.set_title("Frontier theorem statements in the frozen LeanDojo PCA space", fontsize=18, pad=12)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=11, ncol=2, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(figures / "statement_space.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "statement_space.pdf", bbox_inches="tight")
    plt.close(fig)

    # A complementary proof-space view.  Fit the coordinate system only on the
    # frozen LeanDojo reference, then place frontier proofs without refitting.
    # The three spatial axes encode PC1--PC3; color encodes source project and
    # black stars identify the named headline declarations.
    ref_proof, front_proof = arrays["proof"]
    proof_pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    ref_xyz = proof_pca.fit_transform(ref_proof)
    front_xyz = proof_pca.transform(front_proof)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ref_shown = rng.choice(len(ref_xyz), min(5000, len(ref_xyz)), replace=False)
    ax.scatter(
        ref_xyz[ref_shown, 0], ref_xyz[ref_shown, 1], ref_xyz[ref_shown, 2],
        s=3, alpha=0.07, color="#6f6f6f", depthshade=False,
        label="LeanDojo reference",
    )
    for color, project in zip(palette, projects):
        idx = project_indices[project]
        shown = idx if len(idx) <= 500 else rng.choice(idx, 500, replace=False)
        ax.scatter(
            front_xyz[shown, 0], front_xyz[shown, 1], front_xyz[shown, 2],
            s=9, alpha=0.38, color=color, depthshade=False, label=project,
        )
    ax.scatter(
        front_xyz[headline_indices, 0],
        front_xyz[headline_indices, 1],
        front_xyz[headline_indices, 2],
        s=70, marker="*", color="black", depthshade=False, label="headline",
    )
    explained = 100 * proof_pca.explained_variance_ratio_
    ax.set_title("Frontier proofs in the frozen LeanDojo PCA space", fontsize=18, pad=14)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var.)", fontsize=14, labelpad=10)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var.)", fontsize=14, labelpad=10)
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}% var.)", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="z", labelsize=11)
    ax.set_box_aspect((1.25, 1, 0.9))
    ax.view_init(elev=23, azim=-56)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=11, ncol=2, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(figures / "proof_space_3d.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(figures / "proof_space_3d.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # Publication view: equal-sized, vertically stacked 2D panels with a
    # shared encoding and legend.  Each PCA basis is fit only on LeanDojo.
    proof_pca_2d = PCA(n_components=2, svd_solver="randomized", random_state=0)
    ref_proof_xy = proof_pca_2d.fit_transform(ref_proof)
    front_proof_xy = proof_pca_2d.transform(front_proof)
    statement_explained = 100 * pca.explained_variance_ratio_
    proof_explained_2d = 100 * proof_pca_2d.explained_variance_ratio_

    fig = plt.figure(figsize=(9, 9))
    statement_ax = fig.add_axes((0.10, 0.64, 0.84, 0.29))
    proof_ax = fig.add_axes((0.10, 0.23, 0.84, 0.29))

    statement_ax.scatter(
        ref_xy[:, 0], ref_xy[:, 1], s=3, alpha=0.12,
        color="#808080", label="LeanDojo reference",
    )
    for color, project in zip(palette, projects):
        idx = project_indices[project]
        shown = idx if len(idx) <= 600 else rng.choice(idx, 600, replace=False)
        statement_ax.scatter(
            front_xy[shown, 0], front_xy[shown, 1], s=9,
            alpha=0.45, color=color, label=project,
        )
    statement_ax.scatter(
        front_xy[headline_indices, 0], front_xy[headline_indices, 1],
        s=80, marker="*", color="black", label="headline",
    )
    statement_ax.set_title("(a) Statement space", fontsize=18, pad=8)
    statement_ax.set_xlabel(f"PC1 ({statement_explained[0]:.1f}%)", fontsize=14)
    statement_ax.set_ylabel(f"PC2 ({statement_explained[1]:.1f}%)", fontsize=14)
    statement_ax.tick_params(axis="both", labelsize=12)

    proof_ax.scatter(
        ref_proof_xy[:, 0], ref_proof_xy[:, 1],
        s=3, alpha=0.10, color="#6f6f6f",
    )
    for color, project in zip(palette, projects):
        idx = project_indices[project]
        shown = idx if len(idx) <= 600 else rng.choice(idx, 600, replace=False)
        proof_ax.scatter(
            front_proof_xy[shown, 0], front_proof_xy[shown, 1],
            s=9, alpha=0.42, color=color,
        )
    proof_ax.scatter(
        front_proof_xy[headline_indices, 0],
        front_proof_xy[headline_indices, 1],
        s=80, marker="*", color="black",
    )
    proof_ax.set_title("(b) Proof space", fontsize=18, pad=8)
    proof_ax.set_xlabel(f"PC1 ({proof_explained_2d[0]:.1f}%)", fontsize=14)
    proof_ax.set_ylabel(f"PC2 ({proof_explained_2d[1]:.1f}%)", fontsize=14)
    proof_ax.tick_params(axis="both", labelsize=12)

    handles, labels = statement_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=6, fontsize=11,
        markerscale=1.5, frameon=True, bbox_to_anchor=(0.5, 0.015),
    )
    fig.savefig(figures / "frontier_spaces_dual.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(figures / "frontier_spaces_dual.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    lines = [
        "# Results: mapping frontier formalizations", "",
        f"The frozen comparison contains **{len(manifest):,}** frontier declarations and **{len(headlines)}** named headline declarations against **{len(reference_manifest):,}** LeanDojo references.", "",
        "## Main findings", "",
        f"- Frontier statements have mean top-10 LeanDojo similarity **{result['overall']['statement_top10_mean']['mean']:.4f}**, compared with **{result['reference_baselines']['statement_top10_mean']['mean']:.4f}** for LeanDojo declarations against the same reference corpus.",
        f"- **{100 * result['overall']['statement_below_reference_q05_fraction']:.1f}%** of frontier statements fall below the fifth percentile of LeanDojo's own local statement density.",
        f"- Frontier proofs have mean top-10 similarity **{result['overall']['proof_top10_mean']['mean']:.4f}**, compared with **{result['reference_baselines']['proof_top10_mean']['mean']:.4f}** in the reference corpus; **{100 * result['overall']['proof_below_reference_q05_fraction']:.1f}%** fall below its fifth-percentile threshold.",
        f"- Statement neighborhoods still transfer to proof space: proof similarity along statement-selected neighbors exceeds the random-reference baseline by **{result['overall']['transfer_delta']['mean']:.4f}** on average.", "",
        "## Project summary", "",
        "| Project | n | Statement top-10 | Below ref. q05 | Proof top-10 | Transfer delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for project in projects:
        p = project_summaries[project]
        lines.append(
            f"| {project} | {p['n']:,} | {p['statement_top10_mean']['mean']:.4f} | "
            f"{100 * p['statement_below_reference_q05_fraction']:.1f}% | "
            f"{p['proof_top10_mean']['mean']:.4f} | {p['transfer_delta']['mean']:.4f} |"
        )
    lines.extend([
        "", "The statement view is the primary cross-version result. Proof bodies were extracted statically from source and are not equivalent to LeanDojo tactic traces; their comparisons are therefore descriptive.", "",
        "![Statement and proof PCA spaces](figures/frontier_spaces_dual.png)", "",
        "Machine-readable per-project and headline results are in `artifacts/analysis.json`.",
    ])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
