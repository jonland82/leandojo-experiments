"""Measure whether local statement similarity predicts local proof similarity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def coarse_module(file_path: str, components: int) -> str:
    return "/".join(file_path.replace("\\", "/").split("/")[:components])


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains a zero vector")
    return array / norms


def exact_top_neighbors(
    vectors: np.ndarray, maximum_k: int, block_size: int, label: str
) -> tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    indices = np.empty((n, maximum_k), dtype=np.int32)
    similarities = np.empty((n, maximum_k), dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = vectors[start:stop] @ vectors.T
        rows = np.arange(stop - start)
        block[rows, np.arange(start, stop)] = -np.inf
        candidate = np.argpartition(block, -maximum_k, axis=1)[:, -maximum_k:]
        candidate_similarity = np.take_along_axis(block, candidate, axis=1)
        order = np.argsort(-candidate_similarity, axis=1)
        indices[start:stop] = np.take_along_axis(candidate, order, axis=1)
        similarities[start:stop] = np.take_along_axis(
            candidate_similarity, order, axis=1
        )
        print(f"{label} neighbors: {stop}/{n}", flush=True)
    return indices, similarities


def context_matched_random_neighbors(
    reference: np.ndarray,
    module_codes: np.ndarray,
    file_codes: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Match same-file, same-module/different-file, or different-module status."""
    rng = np.random.default_rng(seed)
    n, maximum_k = reference.shape
    output = np.empty_like(reference)
    module_groups = {
        code: np.flatnonzero(module_codes == code)
        for code in np.unique(module_codes)
    }
    file_groups = {
        code: np.flatnonzero(file_codes == code) for code in np.unique(file_codes)
    }
    all_indices = np.arange(n)
    for i in range(n):
        same_file = file_codes[reference[i]] == file_codes[i]
        same_module = module_codes[reference[i]] == module_codes[i]
        module_only = same_module & ~same_file
        different_module = ~same_module
        if np.any(same_file):
            pool = file_groups[file_codes[i]]
            pool = pool[pool != i]
            if len(pool) == 0:
                raise ValueError("Cannot construct a same-file matched baseline")
            output[i, same_file] = rng.choice(
                pool, size=int(same_file.sum()), replace=True
            )
        if np.any(module_only):
            pool = module_groups[module_codes[i]]
            pool = pool[(file_codes[pool] != file_codes[i]) & (pool != i)]
            if len(pool) == 0:
                raise ValueError("Cannot construct a same-module/different-file baseline")
            output[i, module_only] = rng.choice(
                pool, size=int(module_only.sum()), replace=True
            )
        if np.any(different_module):
            draws = rng.choice(
                all_indices, size=int(different_module.sum()), replace=True
            )
            invalid = (draws == i) | (module_codes[draws] == module_codes[i])
            while np.any(invalid):
                draws[invalid] = rng.choice(all_indices, size=int(invalid.sum()), replace=True)
                invalid = (draws == i) | (module_codes[draws] == module_codes[i])
            output[i, different_module] = draws
    return output


def paired_similarities(
    vectors: np.ndarray, neighbors: np.ndarray, block_size: int
) -> np.ndarray:
    output = np.empty(neighbors.shape, dtype=np.float32)
    for start in range(0, len(vectors), block_size):
        stop = min(start + block_size, len(vectors))
        targets = vectors[neighbors[start:stop]]
        output[start:stop] = np.einsum(
            "bd,bkd->bk", vectors[start:stop], targets, optimize=True
        )
    return output


def sampled_pair_similarities(
    vectors: np.ndarray, left: np.ndarray, right: np.ndarray, block_size: int
) -> np.ndarray:
    output = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block_size):
        stop = min(start + block_size, len(left))
        output[start:stop] = np.einsum(
            "bd,bd->b",
            vectors[left[start:stop]],
            vectors[right[start:stop]],
            optimize=True,
        )
    return output


def bootstrap_interval(
    values: np.ndarray, resamples: int, confidence: float, rng: np.random.Generator
) -> tuple[float, float]:
    means = np.empty(resamples, dtype=np.float64)
    block = 50
    for start in range(0, resamples, block):
        stop = min(start + block, resamples)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(means, [tail, 1.0 - tail]))


def neighborhood_overlap(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    overlap = np.empty(len(left), dtype=np.float64)
    for i in range(len(left)):
        overlap[i] = len(set(left[i, :k]).intersection(right[i, :k])) / k
    return overlap


def controlled_regression(
    statement_similarity: np.ndarray,
    proof_similarity: np.ndarray,
    same_module: np.ndarray,
    same_file: np.ndarray,
    tactic_difference: np.ndarray,
) -> dict:
    raw = np.column_stack(
        [statement_similarity, same_module, same_file, tactic_difference]
    ).astype(np.float64)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    standardized = (raw - means) / scales
    y = (proof_similarity - proof_similarity.mean()) / proof_similarity.std()
    design = np.column_stack([np.ones(len(y)), standardized])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    full_r_squared = 1.0 - float(np.sum((y - fitted) ** 2) / np.sum(y**2))
    reduced = np.column_stack([np.ones(len(y)), standardized[:, 1:]])
    reduced_coefficients = np.linalg.lstsq(reduced, y, rcond=None)[0]
    reduced_fitted = reduced @ reduced_coefficients
    reduced_r_squared = 1.0 - float(
        np.sum((y - reduced_fitted) ** 2) / np.sum(y**2)
    )
    names = [
        "intercept",
        "statement_cosine",
        "same_coarse_module",
        "same_source_file",
        "absolute_log1p_tactic_count_difference",
    ]
    return {
        "description": "OLS with standardized response and standardized predictors; descriptive because sampled pairs share theorem endpoints",
        "n_pairs": int(len(y)),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
        "full_r_squared": full_r_squared,
        "r_squared_without_statement_cosine": reduced_r_squared,
        "incremental_r_squared_from_statement_cosine": full_r_squared
        - reduced_r_squared,
    }


def render_results(analysis: dict) -> str:
    lines = [
        "# Do similar Lean statements have similar recorded proofs?",
        "",
        "## Summary",
        "",
        "This experiment tests local cross-view geometry on the same 10,000 theorem--proof pairs used in the semantic-embedding note. It does not call AWS and does not refit the embeddings. For each theorem, it asks whether nearby statements have proofs that are closer than an appropriate random baseline in proof-embedding space.",
        "",
    ]
    k10 = next(row for row in analysis["neighborhood_results"] if row["k"] == 10)
    sign = "yes, with a clear but incomplete association" if k10["proof_cosine_delta"] > 0 else "not detectably positive"
    lines.extend(
        [
            f"The answer is **{sign}**. At `k=10`, proofs attached to a theorem's ten nearest statement neighbors have mean cosine similarity **{k10['mean_proof_cosine_statement_neighbors']:.4f}**, versus **{k10['mean_proof_cosine_context_matched_random']:.4f}** for random neighbors matched as same source file, same coarse module but different file, or different coarse module. The difference is **{k10['proof_cosine_delta']:.4f}** (query-theorem bootstrap 95% interval **[{k10['proof_cosine_delta_ci95'][0]:.4f}, {k10['proof_cosine_delta_ci95'][1]:.4f}]**).",
            "",
            f"The top-10 statement and proof neighborhoods overlap by **{k10['mean_neighborhood_overlap']:.4f}** on average, compared with random expectation **{k10['chance_neighborhood_overlap']:.4f}**; **{100*k10['share_queries_with_any_overlap']:.1f}%** of theorems have at least one neighbor in both top-10 lists. Similar statements therefore make similar recorded proofs appreciably more likely, but do not determine them.",
            "",
            "This is a local result, not a claim that statements determine proofs. Each theorem still contributes only one recorded proof, and nearby statements can admit very different valid arguments.",
            "",
            "## Design",
            "",
            "Let `S_i(k)` and `P_i(k)` be theorem `i`'s exact top-`k` neighbors in statement and proof cosine geometry. For each `k`, the analysis reports:",
            "",
            "- the mean proof cosine along statement-neighbor pairs;",
            "- a random baseline that preserves whether each pair is from the same source file, the same coarse module but different files, or different coarse modules;",
            "- the difference between those quantities, with a theorem-level bootstrap interval; and",
            "- `|S_i(k) intersect P_i(k)| / k`, the fraction of neighbors shared by the two views.",
            "",
            "Exact top-100 neighborhoods are computed from the full 1,024-dimensional matrices. A separate fixed sample of theorem pairs measures global rank correlation and supports a descriptive regression controlling for coarse module, exact source file, and proof-length difference.",
            "",
            "## Local neighborhood results",
            "",
            "| k | Proof cosine: statement neighbors | Context-matched random | Difference | 95% interval | Neighborhood overlap | Chance overlap | Overlap / chance |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["neighborhood_results"]:
        lines.append(
            f"| {row['k']} | {row['mean_proof_cosine_statement_neighbors']:.4f} | "
            f"{row['mean_proof_cosine_context_matched_random']:.4f} | "
            f"{row['proof_cosine_delta']:.4f} | "
            f"[{row['proof_cosine_delta_ci95'][0]:.4f}, {row['proof_cosine_delta_ci95'][1]:.4f}] | "
            f"{row['mean_neighborhood_overlap']:.4f} | {row['chance_neighborhood_overlap']:.4f} | "
            f"{row['overlap_enrichment_over_chance']:.1f}x |"
        )
    correlations = analysis["sampled_pair_results"]["correlations"]
    regression = analysis["sampled_pair_results"]["controlled_regression"]
    lines.extend(
        [
            "",
            "The overlap statistic is deliberately strict: it requires the same theorem to appear among the top `k` neighbors in both spaces. Its random expectation is `k/(n-1)`.",
            "",
            "### Pair-composition check at k=10",
            "",
            f"Among top-10 statement-neighbor pairs, **{100*k10['statement_neighbor_same_coarse_module_share']:.1f}%** share a coarse module and **{100*k10['statement_neighbor_same_source_file_share']:.1f}%** share an exact source file; the random baseline matches those categories pair by pair. Exact statement text occurs in only **{100*k10['statement_neighbor_identical_statement_text_share']:.3f}%** of pairs. Exact recorded proof text occurs in **{100*k10['statement_neighbor_identical_proof_text_share']:.2f}%**, versus **{100*k10['context_matched_random_identical_proof_text_share']:.2f}%** in the matched baseline, so repeated scripts contribute to the effect but cannot by themselves explain it.",
            "",
            "The bootstrap interval treats the query theorem as the resampling unit. Because theorems can also recur as neighbors, the interval is descriptive rather than a fully independent-pair confidence interval.",
            "",
            "## Global pairwise association and controls",
            "",
            f"Across {analysis['sampled_pair_results']['n_pairs']:,} fixed random theorem pairs, statement and proof cosine similarity have Pearson correlation **{correlations['pearson']:.4f}** and Spearman rank correlation **{correlations['spearman']:.4f}**. Pair observations are not independent, so these are effect sizes rather than conventional independent-sample significance tests.",
            "",
            f"In the descriptive standardized regression, the coefficient of statement cosine is **{regression['standardized_coefficients']['statement_cosine']:.4f}** after controlling for shared coarse module, shared source file, and absolute log tactic-count difference. Adding statement cosine increases `R^2` by **{regression['incremental_r_squared_from_statement_cosine']:.4f}**.",
            "",
            "### Proof similarity across statement-similarity deciles",
            "",
            "| Statement-similarity decile | Mean statement cosine | Mean proof cosine | Pairs |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["sampled_pair_results"]["statement_similarity_deciles"]:
        lines.append(
            f"| {row['decile']} | {row['mean_statement_cosine']:.4f} | "
            f"{row['mean_proof_cosine']:.4f} | {row['n_pairs']:,} |"
        )
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            "The cluster result and the neighborhood result answer different questions. Low statement--proof cluster AMI says the two views do not induce the same global taxonomy. The positive local effect here says that unusually similar statements nevertheless carry some information about how their recorded proofs look along the proof view's own axes. Global partition mismatch and local predictability can therefore coexist.",
            "",
            "The context-matched baseline makes the result harder to explain solely by nearby statements sharing a branch of Mathlib or an exact source file. The controlled regression asks the same question across a broad random-pair sample. Neither control removes all leakage: theorem names, reused identifiers, repeated proof scripts, local author conventions, and automation may still contribute.",
            "",
            "Most importantly, this is observational and one-proof-per-theorem. It does not show that a similar theorem must have a similar proof, that the observed proof is canonical, or that similarity will improve proof generation. It supplies the empirical premise for the paid retrieval experiment documented in `AWS_PROOF_GENERATION_PLAN.md`.",
            "",
            "## Reproduction and provenance",
            "",
            f"- Records: **{analysis['n_records']:,}**",
            f"- Dimensions per view: **{analysis['dimensions']:,}**",
            f"- Pair sample seed: **{analysis['config']['seed']}**",
            f"- Random theorem pairs: **{analysis['sampled_pair_results']['n_pairs']:,}**",
            f"- Runtime: **{analysis['elapsed_seconds']:.1f} seconds**",
            f"- Statement embedding SHA-256: `{analysis['inputs']['statement_embedding_sha256']}`",
            f"- Proof embedding SHA-256: `{analysis['inputs']['proof_embedding_sha256']}`",
            f"- Manifest SHA-256: `{analysis['inputs']['manifest_sha256']}`",
            "",
            "Run from the repository root:",
            "",
            "```powershell",
            "python experiments/semantic-neighborhood-transfer-10000/scripts/analyze_geometry.py",
            "```",
            "",
            "The saved top-100 neighbor arrays can be reused as the retrieval candidates for the future generation experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    source = REPO / config["source_experiment"]
    statement_path = source / "artifacts/embeddings/statement.npy"
    proof_path = source / "artifacts/embeddings/proof.npy"
    manifest_path = source / "inputs/manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    statement = normalize_rows(np.load(statement_path, allow_pickle=False))
    proof = normalize_rows(np.load(proof_path, allow_pickle=False))
    if statement.shape != proof.shape or len(manifest) != len(statement):
        raise ValueError("Statement, proof, and manifest sizes do not align")
    n, dimensions = statement.shape
    neighbor_config = config["neighbors"]
    maximum_k = neighbor_config["maximum_k"]
    if maximum_k != max(neighbor_config["k_values"]):
        raise ValueError("maximum_k must equal the largest reported k")

    statement_neighbors, statement_neighbor_cosines = exact_top_neighbors(
        statement,
        maximum_k,
        neighbor_config["matrix_block_size"],
        "statement",
    )
    proof_neighbors, proof_neighbor_cosines = exact_top_neighbors(
        proof,
        maximum_k,
        neighbor_config["matrix_block_size"],
        "proof",
    )

    modules = np.asarray(
        [
            coarse_module(
                row["file_path"], config["controls"]["coarse_module_path_components"]
            )
            for row in manifest
        ]
    )
    _, module_codes = np.unique(modules, return_inverse=True)
    file_paths = np.asarray([row["file_path"] for row in manifest])
    _, file_codes = np.unique(file_paths, return_inverse=True)
    matched_neighbors = context_matched_random_neighbors(
        statement_neighbors, module_codes, file_codes, config["seed"] + 101
    )
    row_indices = np.arange(n)[:, None]
    if not np.array_equal(
        file_codes[statement_neighbors] == file_codes[row_indices],
        file_codes[matched_neighbors] == file_codes[row_indices],
    ):
        raise AssertionError("Matched baseline did not preserve same-file status")
    if not np.array_equal(
        module_codes[statement_neighbors] == module_codes[row_indices],
        module_codes[matched_neighbors] == module_codes[row_indices],
    ):
        raise AssertionError("Matched baseline did not preserve same-module status")
    proof_cosines_along_statement = paired_similarities(
        proof,
        statement_neighbors,
        neighbor_config["paired_similarity_block_size"],
    )
    proof_cosines_matched_random = paired_similarities(
        proof,
        matched_neighbors,
        neighbor_config["paired_similarity_block_size"],
    )
    statement_hashes = np.asarray([row["statement_sha256"] for row in manifest])
    proof_hashes = np.asarray([row["proof_sha256"] for row in manifest])

    bootstrap_rng = np.random.default_rng(config["seed"] + 202)
    neighborhood_results = []
    for k in neighbor_config["k_values"]:
        observed_by_query = proof_cosines_along_statement[:, :k].mean(axis=1)
        baseline_by_query = proof_cosines_matched_random[:, :k].mean(axis=1)
        difference_by_query = observed_by_query - baseline_by_query
        difference_interval = bootstrap_interval(
            difference_by_query,
            config["inference"]["bootstrap_resamples"],
            config["inference"]["confidence_level"],
            bootstrap_rng,
        )
        overlap_by_query = neighborhood_overlap(
            statement_neighbors, proof_neighbors, k
        )
        overlap_interval = bootstrap_interval(
            overlap_by_query,
            config["inference"]["bootstrap_resamples"],
            config["inference"]["confidence_level"],
            bootstrap_rng,
        )
        chance_overlap = k / (n - 1)
        mean_overlap = float(overlap_by_query.mean())
        neighborhood_results.append(
            {
                "k": k,
                "mean_statement_neighbor_cosine": float(
                    statement_neighbor_cosines[:, :k].mean()
                ),
                "mean_proof_cosine_statement_neighbors": float(
                    observed_by_query.mean()
                ),
                "mean_proof_cosine_context_matched_random": float(
                    baseline_by_query.mean()
                ),
                "proof_cosine_delta": float(difference_by_query.mean()),
                "proof_cosine_delta_ci95": list(difference_interval),
                "mean_neighborhood_overlap": mean_overlap,
                "neighborhood_overlap_ci95": list(overlap_interval),
                "chance_neighborhood_overlap": chance_overlap,
                "overlap_enrichment_over_chance": mean_overlap / chance_overlap,
                "share_queries_with_any_overlap": float(
                    np.mean(overlap_by_query > 0)
                ),
                "statement_neighbor_same_coarse_module_share": float(
                    np.mean(module_codes[statement_neighbors[:, :k]] == module_codes[row_indices])
                ),
                "statement_neighbor_same_source_file_share": float(
                    np.mean(file_codes[statement_neighbors[:, :k]] == file_codes[row_indices])
                ),
                "statement_neighbor_identical_statement_text_share": float(
                    np.mean(statement_hashes[statement_neighbors[:, :k]] == statement_hashes[row_indices])
                ),
                "statement_neighbor_identical_proof_text_share": float(
                    np.mean(proof_hashes[statement_neighbors[:, :k]] == proof_hashes[row_indices])
                ),
                "context_matched_random_identical_proof_text_share": float(
                    np.mean(proof_hashes[matched_neighbors[:, :k]] == proof_hashes[row_indices])
                ),
            }
        )

    pair_config = config["random_pairs"]
    pair_rng = np.random.default_rng(config["seed"] + 303)
    left = pair_rng.integers(0, n, size=pair_config["sample_size"], dtype=np.int32)
    right = pair_rng.integers(0, n, size=pair_config["sample_size"], dtype=np.int32)
    equal = left == right
    while np.any(equal):
        right[equal] = pair_rng.integers(0, n, size=int(equal.sum()), dtype=np.int32)
        equal = left == right
    statement_pair_cosine = sampled_pair_similarities(
        statement, left, right, pair_config["similarity_block_size"]
    )
    proof_pair_cosine = sampled_pair_similarities(
        proof, left, right, pair_config["similarity_block_size"]
    )
    same_module = (module_codes[left] == module_codes[right]).astype(np.float64)
    same_file = (file_paths[left] == file_paths[right]).astype(np.float64)
    tactic_counts = np.asarray([row["n_tactics"] for row in manifest], dtype=np.float64)
    tactic_difference = np.abs(
        np.log1p(tactic_counts[left]) - np.log1p(tactic_counts[right])
    )
    pearson = float(pearsonr(statement_pair_cosine, proof_pair_cosine).statistic)
    spearman = float(spearmanr(statement_pair_cosine, proof_pair_cosine).statistic)
    regression = controlled_regression(
        statement_pair_cosine,
        proof_pair_cosine,
        same_module,
        same_file,
        tactic_difference,
    )
    edges = np.quantile(statement_pair_cosine, np.linspace(0, 1, 11))
    bins = np.searchsorted(edges[1:-1], statement_pair_cosine, side="right")
    deciles = []
    for decile in range(10):
        mask = bins == decile
        deciles.append(
            {
                "decile": decile + 1,
                "n_pairs": int(mask.sum()),
                "minimum_statement_cosine": float(statement_pair_cosine[mask].min()),
                "maximum_statement_cosine": float(statement_pair_cosine[mask].max()),
                "mean_statement_cosine": float(statement_pair_cosine[mask].mean()),
                "mean_proof_cosine": float(proof_pair_cosine[mask].mean()),
            }
        )

    artifacts = EXPERIMENT / "artifacts"
    save_array(artifacts / "statement_top100_indices.npy", statement_neighbors)
    save_array(artifacts / "statement_top100_cosine.npy", statement_neighbor_cosines)
    save_array(artifacts / "proof_top100_indices.npy", proof_neighbors)
    save_array(artifacts / "proof_top100_cosine.npy", proof_neighbor_cosines)
    analysis = {
        "experiment_id": config["experiment_id"],
        "question": "Do similar theorem statements have similar recorded proofs along proof-embedding axes?",
        "config": config,
        "n_records": n,
        "dimensions": dimensions,
        "inputs": {
            "statement_embedding_path": str(statement_path.relative_to(REPO)),
            "statement_embedding_sha256": sha256_file(statement_path),
            "proof_embedding_path": str(proof_path.relative_to(REPO)),
            "proof_embedding_sha256": sha256_file(proof_path),
            "manifest_path": str(manifest_path.relative_to(REPO)),
            "manifest_sha256": sha256_file(manifest_path),
            "split_counts": dict(Counter(row["source_split"] for row in manifest)),
        },
        "neighborhood_results": neighborhood_results,
        "sampled_pair_results": {
            "n_pairs": int(len(left)),
            "correlations": {"pearson": pearson, "spearman": spearman},
            "same_coarse_module_share": float(same_module.mean()),
            "same_source_file_share": float(same_file.mean()),
            "controlled_regression": regression,
            "statement_similarity_deciles": deciles,
        },
        "neighbor_artifacts": {},
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    for name in [
        "statement_top100_indices.npy",
        "statement_top100_cosine.npy",
        "proof_top100_indices.npy",
        "proof_top100_cosine.npy",
    ]:
        path = artifacts / name
        analysis["neighbor_artifacts"][name] = {
            "shape": list(np.load(path, mmap_mode="r", allow_pickle=False).shape),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(artifacts / "analysis.json", analysis)
    (EXPERIMENT / "RESULTS.md").write_text(
        render_results(analysis), encoding="utf-8"
    )
    print(json.dumps({
        "elapsed_seconds": analysis["elapsed_seconds"],
        "pearson": pearson,
        "spearman": spearman,
        "k10": next(row for row in neighborhood_results if row["k"] == 10),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
