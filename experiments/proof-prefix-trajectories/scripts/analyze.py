"""Measure within-proof semantic trajectories against the frozen 10k corpus."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding")
    return array / norms


def effective(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if len(positive) == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def bootstrap_median_ci(values: np.ndarray, repetitions: int, rng: np.random.RandomState) -> list[float]:
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return [float(x) for x in np.percentile(medians, [2.5, 97.5])]


def paired_summary(
    start: np.ndarray,
    end: np.ndarray,
    predicted: str,
    repetitions: int,
    rng: np.random.RandomState,
) -> dict:
    delta = end - start
    try:
        test = wilcoxon(delta)
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    except ValueError:
        statistic, pvalue = 0.0, 1.0
    favorable = delta < 0 if predicted == "decrease" else delta > 0
    return {
        "start_median": float(np.median(start)),
        "end_median": float(np.median(end)),
        "median_change": float(np.median(delta)),
        "median_change_ci95": bootstrap_median_ci(delta, repetitions, rng),
        "share_in_predicted_direction": float(np.mean(favorable)),
        "share_unchanged": float(np.mean(delta == 0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_two_sided_p": pvalue,
    }


def main() -> None:
    input_path = EXPERIMENT / "inputs/trajectories.jsonl"
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    trajectory_path = EXPERIMENT / "artifacts/embeddings.npy"
    vectors = normalize(np.load(trajectory_path, allow_pickle=False).astype(np.float64))
    source = REPO / CONFIG["source_experiment"]
    reference_path = source / "artifacts/embeddings/proof.npy"
    reference = normalize(np.load(reference_path, allow_pickle=False).astype(np.float64))
    if vectors.shape != (len(rows), reference.shape[1]):
        raise RuntimeError(f"unexpected trajectory shape {vectors.shape}")

    top_k = CONFIG["analysis"]["top_neighbors"]
    tau = CONFIG["analysis"]["cosine_threshold"]
    by_sample: dict[int, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_sample[row["sample_i"]].append(row_index)

    initial_geometry: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sample_i, indices in by_sample.items():
        initial_index = next(
            index
            for index in indices
            if rows[index]["trajectory"] == "actual" and rows[index]["checkpoint"] == 1
        )
        target_i = rows[initial_index]["target_i"]
        similarities = reference @ vectors[initial_index]
        similarities[target_i] = -np.inf
        neighbor_indices = np.argpartition(similarities, -top_k)[-top_k:]
        neighbor_indices = neighbor_indices[np.argsort(similarities[neighbor_indices])[::-1]]
        neighbor_vectors = reference[neighbor_indices]
        basis, _ = np.linalg.qr(neighbor_vectors.T, mode="reduced")
        initial_geometry[sample_i] = (
            neighbor_indices,
            neighbor_vectors,
            similarities[neighbor_indices].copy(),
            basis,
        )

    score_rows: list[dict] = []
    for start in range(0, len(rows), 128):
        end = min(len(rows), start + 128)
        similarities = vectors[start:end] @ reference.T
        for local, row_index in enumerate(range(start, end)):
            row = rows[row_index]
            sims = similarities[local]
            sims[row["target_i"]] = -np.inf
            selected = np.argpartition(sims, -top_k)[-top_k:]
            selected_sims = np.sort(sims[selected])[::-1]
            weights = np.maximum(0.0, selected_sims - tau)
            initial_indices, neighbor_vectors, initial_sims, basis = initial_geometry[row["sample_i"]]
            frozen_sims = neighbor_vectors @ vectors[row_index]
            frozen_weights = np.maximum(0.0, frozen_sims - tau)
            attenuation = float(np.dot(initial_sims, frozen_sims) / np.dot(initial_sims, initial_sims))
            fit_residual = frozen_sims - attenuation * initial_sims
            projection_energy = float(np.sum((basis.T @ vectors[row_index]) ** 2))
            score_rows.append(
                {
                    key: row[key]
                    for key in (
                        "input_i", "sample_i", "target_i", "full_name", "file_path",
                        "trajectory", "checkpoint", "total_tactics", "is_full",
                    )
                }
                | {
                    "weighted_degree": float(weights.sum()),
                    "effective_neighbors": effective(weights),
                    "neighbor_count": int(np.count_nonzero(weights)),
                    "retained_significant_neighbor_count": int(
                        np.count_nonzero((sims[initial_indices] > tau))
                    ),
                    "new_significant_neighbor_count": int(
                        np.count_nonzero(weights)
                        - np.count_nonzero((sims[initial_indices] > tau))
                    ),
                    "maximum_similarity": float(selected_sims[0]),
                    "initial_weighted_degree": float(frozen_weights.sum()),
                    "initial_effective_neighbors": effective(frozen_weights),
                    "initial_neighbor_count": int(np.count_nonzero(frozen_weights)),
                    "initial_neighbor_mean_similarity": float(np.mean(frozen_sims)),
                    "initial_neighbor_attenuation": attenuation,
                    "attenuation_fit_rmse": float(np.sqrt(np.mean(fit_residual ** 2))),
                    "initial_neighbor_subspace_residual": float(max(0.0, 1.0 - projection_energy)),
                }
            )
    scores_path = EXPERIMENT / "artifacts/scores.jsonl"
    write_jsonl(scores_path, score_rows)

    lookup = {
        (row["sample_i"], row["trajectory"], row["checkpoint"]): row
        for row in score_rows
    }
    fixed_checkpoints = CONFIG["sample"]["checkpoints"]
    metrics = (
        "weighted_degree",
        "effective_neighbors",
        "neighbor_count",
        "retained_significant_neighbor_count",
        "new_significant_neighbor_count",
        "maximum_similarity",
        "initial_weighted_degree",
        "initial_effective_neighbors",
        "initial_neighbor_count",
        "initial_neighbor_mean_similarity",
        "initial_neighbor_attenuation",
        "attenuation_fit_rmse",
        "initial_neighbor_subspace_residual",
    )
    checkpoint_summary: dict[str, dict] = {}
    for trajectory in ("actual", "repeat"):
        checkpoint_summary[trajectory] = {}
        for checkpoint in fixed_checkpoints:
            selected = [
                lookup[(sample_i, trajectory, checkpoint)] for sample_i in sorted(by_sample)
            ]
            checkpoint_summary[trajectory][str(checkpoint)] = {
                metric: {
                    "median": float(np.median([row[metric] for row in selected])),
                    "mean": float(np.mean([row[metric] for row in selected])),
                }
                for metric in metrics
            }

    primary = CONFIG["analysis"]["primary_checkpoint"]
    repetitions = CONFIG["analysis"]["bootstrap_repetitions"]
    rng = np.random.RandomState(CONFIG["analysis"]["bootstrap_seed"])
    direction = {
        "weighted_degree": "decrease",
        "effective_neighbors": "decrease",
        "neighbor_count": "decrease",
        "retained_significant_neighbor_count": "decrease",
        "new_significant_neighbor_count": "increase",
        "maximum_similarity": "decrease",
        "initial_weighted_degree": "decrease",
        "initial_effective_neighbors": "decrease",
        "initial_neighbor_count": "decrease",
        "initial_neighbor_mean_similarity": "decrease",
        "initial_neighbor_attenuation": "decrease",
        "attenuation_fit_rmse": "increase",
        "initial_neighbor_subspace_residual": "increase",
    }
    primary_changes: dict[str, dict] = {}
    deltas: dict[tuple[str, str], np.ndarray] = {}
    for trajectory in ("actual", "repeat"):
        primary_changes[trajectory] = {}
        for metric in metrics:
            start_values = np.asarray(
                [lookup[(sample_i, trajectory, 1)][metric] for sample_i in sorted(by_sample)]
            )
            end_values = np.asarray(
                [lookup[(sample_i, trajectory, primary)][metric] for sample_i in sorted(by_sample)]
            )
            deltas[(trajectory, metric)] = end_values - start_values
            primary_changes[trajectory][metric] = paired_summary(
                start_values, end_values, direction[metric], repetitions, rng
            )

    difference_in_differences: dict[str, dict] = {}
    for metric in metrics:
        delta = deltas[("actual", metric)] - deltas[("repeat", metric)]
        predicted = direction[metric]
        try:
            test = wilcoxon(delta)
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, pvalue = 0.0, 1.0
        difference_in_differences[metric] = {
            "median_actual_minus_repeat_change": float(np.median(delta)),
            "ci95": bootstrap_median_ci(delta, repetitions, rng),
            "share_actual_more_predicted_than_repeat": float(
                np.mean(delta < 0 if predicted == "decrease" else delta > 0)
            ),
            "wilcoxon_statistic": statistic,
            "wilcoxon_two_sided_p": pvalue,
        }

    residual_delta = deltas[("actual", "initial_neighbor_subspace_residual")]
    mechanism_associations = {}
    for metric in (
        "weighted_degree", "effective_neighbors", "neighbor_count",
        "initial_weighted_degree", "initial_effective_neighbors", "initial_neighbor_count",
    ):
        result = spearmanr(residual_delta, deltas[("actual", metric)])
        mechanism_associations[f"residual_change_vs_{metric}_change"] = {
            "spearman_rho": float(result.statistic),
            "two_sided_p": float(result.pvalue),
        }

    monotonicity: dict[str, dict] = {}
    for trajectory in ("actual", "repeat"):
        monotonicity[trajectory] = {}
        for metric in metrics:
            rhos = []
            for sample_i in sorted(by_sample):
                values = [lookup[(sample_i, trajectory, checkpoint)][metric] for checkpoint in fixed_checkpoints]
                rho = (
                    0.0
                    if np.ptp(values) == 0
                    else spearmanr(fixed_checkpoints, values).statistic
                )
                rhos.append(float(rho) if np.isfinite(rho) else 0.0)
            rhos_array = np.asarray(rhos)
            predicted_negative = direction[metric] == "decrease"
            monotonicity[trajectory][metric] = {
                "median_within_proof_spearman": float(np.median(rhos_array)),
                "share_with_predicted_spearman_sign": float(
                    np.mean(rhos_array < 0 if predicted_negative else rhos_array > 0)
                ),
            }

    duplicate_cosines = []
    full_reembedding_cosines = []
    for sample_i in sorted(by_sample):
        actual_one = lookup[(sample_i, "actual", 1)]
        repeat_one = lookup[(sample_i, "repeat", 1)]
        duplicate_cosines.append(
            float(np.dot(vectors[actual_one["input_i"]], vectors[repeat_one["input_i"]]))
        )
        full = next(
            row for row in score_rows
            if row["sample_i"] == sample_i and row["trajectory"] == "actual" and row["is_full"]
        )
        full_reembedding_cosines.append(
            float(np.dot(vectors[full["input_i"]], reference[full["target_i"]]))
        )

    analysis = {
        "experiment_id": CONFIG["experiment_id"],
        "sample_size": len(by_sample),
        "trajectory_inputs": len(rows),
        "reference_proofs": len(reference),
        "threshold": tau,
        "top_neighbors": top_k,
        "checkpoint_summary": checkpoint_summary,
        "primary_change_1_to_16": primary_changes,
        "difference_in_differences": difference_in_differences,
        "mechanism_associations": mechanism_associations,
        "within_proof_monotonicity": monotonicity,
        "reliability": {
            "duplicate_checkpoint_1_cosine_minimum": float(np.min(duplicate_cosines)),
            "duplicate_checkpoint_1_cosine_median": float(np.median(duplicate_cosines)),
            "full_reembedding_to_archived_cosine_minimum": float(np.min(full_reembedding_cosines)),
            "full_reembedding_to_archived_cosine_median": float(np.median(full_reembedding_cosines)),
        },
        "files": {
            "inputs": {"path": str(input_path.relative_to(REPO)), "sha256": sha256_file(input_path)},
            "trajectory_embeddings": {
                "path": str(trajectory_path.relative_to(REPO)), "sha256": sha256_file(trajectory_path)
            },
            "reference_embeddings": {
                "path": str(reference_path.relative_to(REPO)), "sha256": sha256_file(reference_path)
            },
            "scores": {"path": str(scores_path.relative_to(REPO)), "sha256": sha256_file(scores_path)},
        },
    }
    write_json(EXPERIMENT / "artifacts/analysis.json", analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
