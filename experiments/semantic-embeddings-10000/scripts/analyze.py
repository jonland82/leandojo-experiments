"""Cluster the three embedding views and render the standalone result report."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import chi2_contingency
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def fit_cluster(X: np.ndarray, k: int, seed: int, config: dict) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=config["n_init"],
        max_iter=config["max_iter"],
        batch_size=config["batch_size"],
        random_state=seed,
        reassignment_ratio=0.01,
    )
    model.fit(X)
    return model


def cramers_v(left: np.ndarray, right: np.ndarray) -> float:
    left_values, left_codes = np.unique(left, return_inverse=True)
    right_values, right_codes = np.unique(right, return_inverse=True)
    table = np.bincount(
        left_codes * len(right_values) + right_codes,
        minlength=len(left_values) * len(right_values),
    ).reshape(len(left_values), len(right_values))
    chi_squared = chi2_contingency(table, correction=False)[0]
    denominator = table.sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(math.sqrt(chi_squared / denominator)) if denominator else 0.0


def association(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(left), dtype=bool)
    left = left[mask]
    right = right[mask]
    return {
        "n": int(mask.sum()),
        "ami": float(adjusted_mutual_info_score(left, right)),
        "nmi": float(normalized_mutual_info_score(left, right)),
        "cramers_v": cramers_v(left, right),
    }


def coarse_module(file_path: str) -> str:
    return "/".join(file_path.replace("\\", "/").split("/")[:2])


def select_k(diagnostics: list[dict], threshold: float) -> tuple[int, str]:
    stable = [row for row in diagnostics if row["mean_subsample_ami"] >= threshold]
    if stable:
        selected = max(stable, key=lambda row: (row["cosine_silhouette"], -row["k"]))
        return selected["k"], "maximum cosine silhouette among stable candidates"
    selected = max(
        diagnostics,
        key=lambda row: (row["mean_subsample_ami"], row["cosine_silhouette"], -row["k"]),
    )
    return selected["k"], "fallback: maximum mean subsample AMI because no candidate met threshold"


def cluster_summary(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray, manifest: list[dict]
) -> list[dict]:
    normalized_centers = normalize(centers.astype(np.float64), norm="l2")
    summaries: list[dict] = []
    for cluster in range(centers.shape[0]):
        members = np.flatnonzero(labels == cluster)
        similarity = X[members] @ normalized_centers[cluster]
        representative_members = members[np.argsort(-similarity)[:5]]
        modules = Counter(coarse_module(manifest[index]["file_path"]) for index in members)
        tactic_counts = np.asarray([manifest[index]["n_tactics"] for index in members])
        summaries.append(
            {
                "cluster": cluster,
                "size": int(len(members)),
                "share": float(len(members) / len(labels)),
                "mean_tactics": float(tactic_counts.mean()),
                "median_tactics": float(np.median(tactic_counts)),
                "top_modules": [
                    {"module": module, "count": count}
                    for module, count in modules.most_common(5)
                ],
                "representatives": [
                    {
                        "full_name": manifest[index]["full_name"],
                        "file_path": manifest[index]["file_path"],
                        "cosine_to_center": float(
                            X[index] @ normalized_centers[cluster]
                        ),
                    }
                    for index in representative_members
                ],
            }
        )
    return summaries


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(analysis: dict, preparation: dict, embedding: dict, config: dict) -> str:
    views = analysis["views"]
    lines = [
        "# Semantic theorem/proof embeddings at 10,000 proofs",
        "",
        "## Summary",
        "",
    ]
    selected = ", ".join(
        f"{name} **K={payload['selected_k']}**" for name, payload in views.items()
    )
    pair_lookup = {row["comparison"]: row for row in analysis["cross_view_associations"]}
    sp = pair_lookup["statement vs proof"]
    sj = pair_lookup["statement vs joint"]
    pj = pair_lookup["proof vs joint"]
    lines.extend(
        [
            f"The frozen selection rule chose {selected}. Statement-only and proof-only clusters "
            f"have AMI **{format_float(sp['ami'])}** (Cramer's V **{format_float(sp['cramers_v'])}**). "
            "This is the direct semantic-view analogue of the earlier style/domain comparison.",
            "",
            "No candidate in any view met the preregistered-style stability threshold of mean "
            "subsample AMI >= 0.8. The reported K values therefore come from the frozen fallback "
            "rule (maximum observed stability), and should be treated as exploratory resolutions "
            "rather than evidence for natural cluster counts.",
            "",
            f"The joint representation aligns with statement clusters at AMI **{format_float(sj['ami'])}** "
            f"and with proof clusters at AMI **{format_float(pj['ami'])}**. These values indicate which "
            "side of the concatenated representation dominates its geometry.",
            "",
            "The external comparisons provide a useful convergent-validity check: statement embeddings "
            f"align more with the earlier domain topics (AMI **{format_float(next(row['ami'] for row in analysis['legacy_associations'] if row['comparison'] == 'statement vs domain'))}**) "
            f"than style topics (**{format_float(next(row['ami'] for row in analysis['legacy_associations'] if row['comparison'] == 'statement vs style'))}**), while proof embeddings align more with style "
            f"(**{format_float(next(row['ami'] for row in analysis['legacy_associations'] if row['comparison'] == 'proof vs style'))}**) than domain (**{format_float(next(row['ami'] for row in analysis['legacy_associations'] if row['comparison'] == 'proof vs domain'))}**).",
            "",
            "These are empirical code/text embedding clusters, not equivalence classes in Lean's logic. "
            "The model was not trained to prove definitional or propositional equivalence.",
            "",
            "## Isolation and provenance",
            "",
            "This experiment is stored entirely in `experiments/semantic-embeddings-10000/`. It did not "
            "modify `experiments/aws-10000/`, `FINDINGS.md`, `out/`, or `app/data.js`. The earlier artifact "
            "was read only for exact sample-alignment validation and comparison labels.",
            "",
            f"The sample reproduces the seed-{preparation['sample']['seed']} uniform sample of "
            f"{preparation['sample']['selected']:,} records from "
            f"{preparation['sample']['available_tactic_proofs']:,} records with nonempty tactic traces. "
            f"Exact order alignment with the earlier artifact: **{str(preparation['sample']['aligned_with_aws_10000_artifact']).lower()}**.",
            "",
            "The theorem declaration text came from `corpus.jsonl`. Lookup used theorem full name plus a "
            "normalized source-path suffix match; this disambiguates names appearing in more than one corpus file.",
            "",
            "| Split | Available | Selected |",
            "|---|---:|---:|",
        ]
    )
    for split in config["dataset"]["splits"]:
        lines.append(
            f"| {split} | {preparation['sample']['available_by_split'][split]:,} | "
            f"{preparation['sample']['selected_by_split'].get(split, 0):,} |"
        )

    lines.extend(
        [
            "",
            "## Representations and embedding run",
            "",
            "| View | Records | Characters | Vector shape | SHA-256 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, payload in embedding["views"].items():
        lines.append(
            f"| {name} | {payload['inputs']:,} | {payload['characters']:,} | "
            f"{payload['shape'][0]:,} x {payload['shape'][1]:,} | `{payload['sha256'][:16]}…` |"
        )
    cost = embedding["estimated_embedding_cost_usd"]
    lines.extend(
        [
            "",
            f"Embeddings were generated with `{config['embedding']['model_id']}` through the AWS CLI "
            f"command `aws bedrock-runtime invoke-model` in `{config['embedding']['region']}`, using "
            f"`input_type={config['embedding']['input_type']}`, {config['embedding']['output_dimension']} "
            "float dimensions, and no truncation. Inputs were batched in source order and each completed "
            "batch was checkpointed before assembly.",
            "",
            f"Embedding wall time was **{embedding['elapsed_seconds'] / 60:.1f} minutes**. Bedrock did not "
            "return a billing-token total, so cost is estimated from characters: "
            f"**${cost['at_4_characters_per_token']:.2f}–${cost['at_2_5_characters_per_token']:.2f}** "
            f"at the recorded ${config['embedding']['estimated_price_usd_per_million_text_tokens']:.2f}/M-token rate.",
            "",
            "## Clustering protocol",
            "",
            "Each matrix was L2-normalized and clustered independently with MiniBatchKMeans. Candidate "
            "resolutions were K = 4, 6, 8, 10, 12, 14, and 16. For each K, a full-data fit was compared "
            "against four independent 80% subsample fits using adjusted mutual information. The frozen "
            "selection rule chooses the highest cosine silhouette among candidates with mean stability "
            "AMI at least 0.8; if none qualify, it chooses maximum stability.",
            "",
            "### Candidate diagnostics",
            "",
        ]
    )
    for name, payload in views.items():
        lines.extend(
            [
                f"#### {name.capitalize()}",
                "",
                "| K | Cosine silhouette | Mean subsample AMI | Min subsample AMI | Inertia / record | Selected |",
                "|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in payload["diagnostics"]:
            lines.append(
                f"| {row['k']} | {format_float(row['cosine_silhouette'])} | "
                f"{format_float(row['mean_subsample_ami'])} | {format_float(row['minimum_subsample_ami'])} | "
                f"{format_float(row['inertia_per_record'])} | {'yes' if row['k'] == payload['selected_k'] else ''} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Cross-view alignment",
            "",
            "| Comparison | N | AMI | NMI | Cramer's V |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["cross_view_associations"]:
        lines.append(
            f"| {row['comparison']} | {row['n']:,} | {format_float(row['ami'])} | "
            f"{format_float(row['nmi'])} | {format_float(row['cramers_v'])} |"
        )

    lines.extend(
        [
            "",
            "## Alignment with the earlier feature views",
            "",
            "| Semantic view vs earlier view | N | AMI | NMI | Cramer's V |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["legacy_associations"]:
        lines.append(
            f"| {row['comparison']} | {row['n']:,} | {format_float(row['ami'])} | "
            f"{format_float(row['nmi'])} | {format_float(row['cramers_v'])} |"
        )

    lines.extend(
        [
            "",
            "## Alignment with source modules",
            "",
            "| Semantic view | N | AMI | NMI | Cramer's V |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, payload in views.items():
        row = payload["module_association"]
        lines.append(
            f"| {name} | {row['n']:,} | {format_float(row['ami'])} | "
            f"{format_float(row['nmi'])} | {format_float(row['cramers_v'])} |"
        )
    lines.extend(
        [
            "",
            "Statement and joint embeddings align much more strongly with the coarse source hierarchy "
            "than proof embeddings. This reproduces the earlier experiment's asymmetry: mathematical "
            "content is associated with where a theorem lives, while proof procedure travels more freely "
            "across modules.",
        ]
    )

    lines.extend(["", "## Selected cluster summaries", ""])
    for name, payload in views.items():
        lines.extend(
            [
                f"### {name.capitalize()} clusters (K={payload['selected_k']})",
                "",
                "| Cluster | Records | Share | Mean tactics | Leading source modules | Representative theorems |",
                "|---:|---:|---:|---:|---|---|",
            ]
        )
        for cluster in payload["clusters"]:
            modules = ", ".join(
                f"`{item['module']}` ({item['count']})" for item in cluster["top_modules"][:3]
            )
            representatives = ", ".join(
                f"`{item['full_name']}`" for item in cluster["representatives"][:3]
            )
            lines.append(
                f"| {cluster['cluster']} | {cluster['size']:,} | {100 * cluster['share']:.1f}% | "
                f"{cluster['mean_tactics']:.2f} | {modules} | {representatives} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- Cohere Embed v4 is a general code/text embedding model, not a Lean kernel or a model of proof equivalence.",
            "- Cluster labels describe geometry at the selected resolution; they are not a canonical taxonomy.",
            "- The joint representation is a single embedding of concatenated fields, so it does not explicitly balance statement and proof information.",
            "- Tactic proofs expose only the traced source tactics. Automation can invoke internal lemmas not listed in tactic syntax.",
            "- Silhouette values in high-dimensional semantic spaces are often small; stability and external alignment should be considered alongside them.",
            "",
            "## Reproduction",
            "",
            "From the repository root:",
            "",
            "```powershell",
            "python experiments/semantic-embeddings-10000/scripts/prepare_inputs.py",
            "python experiments/semantic-embeddings-10000/scripts/embed_aws_cli.py",
            "python experiments/semantic-embeddings-10000/scripts/analyze.py",
            "```",
            "",
            "The embedding step is resumable: existing valid batch arrays are reused. Exact settings are "
            "frozen in `config.json`; input and embedding checksums are recorded in `artifacts/`.",
            "",
            "## Runtime environment",
            "",
            f"- Python: `{analysis['runtime']['python'].splitlines()[0]}`",
            f"- NumPy: `{analysis['runtime']['numpy']}`",
            f"- SciPy: `{analysis['runtime']['scipy']}`",
            f"- scikit-learn: `{analysis['runtime']['scikit_learn']}`",
            f"- Platform: `{analysis['runtime']['platform']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    cluster_config = config["clustering"]
    preparation = json.loads(
        (EXPERIMENT / "artifacts/preparation.json").read_text(encoding="utf-8")
    )
    embedding = json.loads(
        (EXPERIMENT / "artifacts/embedding_run.json").read_text(encoding="utf-8")
    )
    manifest = load_jsonl(EXPERIMENT / "inputs/manifest.jsonl")
    old_payload = json.loads(
        (REPO / "experiments/aws-10000/artifacts/proofs.json").read_text(encoding="utf-8")
    )
    old_points = old_payload["points"]
    style = np.asarray([point["c"] for point in old_points], dtype=int)
    domain = np.asarray([point["domain_c"] for point in old_points], dtype=int)
    modules = np.asarray([coarse_module(row["file_path"]) for row in manifest])

    view_results: dict[str, dict] = {}
    selected_labels: dict[str, np.ndarray] = {}
    for view in config["views"]:
        raw = np.load(EXPERIMENT / f"artifacts/embeddings/{view}.npy", allow_pickle=False)
        X = normalize(raw.astype(np.float64), norm="l2").astype(np.float32)
        diagnostics: list[dict] = []
        full_models: dict[int, MiniBatchKMeans] = {}
        for k in cluster_config["candidate_k"]:
            fit_started = time.perf_counter()
            full_model = fit_cluster(X, k, cluster_config["seed"], cluster_config)
            full_models[k] = full_model
            full_labels = full_model.labels_
            silhouette = silhouette_score(
                X,
                full_labels,
                metric="cosine",
                sample_size=min(cluster_config["silhouette_sample_size"], len(X)),
                random_state=cluster_config["seed"],
            )
            stability: list[float] = []
            for offset in range(cluster_config["stability_subsamples"]):
                rng = np.random.default_rng(cluster_config["seed"] + offset + 1)
                subset = np.sort(
                    rng.choice(
                        len(X),
                        size=int(len(X) * cluster_config["stability_fraction"]),
                        replace=False,
                    )
                )
                sub_model = fit_cluster(
                    X[subset], k, cluster_config["seed"] + offset + 1, cluster_config
                )
                stability.append(
                    float(adjusted_mutual_info_score(full_labels[subset], sub_model.labels_))
                )
            row = {
                "k": k,
                "cosine_silhouette": float(silhouette),
                "mean_subsample_ami": float(np.mean(stability)),
                "minimum_subsample_ami": float(np.min(stability)),
                "subsample_ami": stability,
                "inertia": float(full_model.inertia_),
                "inertia_per_record": float(full_model.inertia_ / len(X)),
                "fit_and_diagnostics_seconds": time.perf_counter() - fit_started,
            }
            diagnostics.append(row)
            print(
                f"{view} k={k}: silhouette={silhouette:.4f}, "
                f"stability={np.mean(stability):.4f}",
                flush=True,
            )

        selected_k, selection_reason = select_k(
            diagnostics, cluster_config["stability_threshold_ami"]
        )
        model = full_models[selected_k]
        labels = model.labels_.astype(np.int16)
        selected_labels[view] = labels
        label_path = EXPERIMENT / f"artifacts/clusters/{view}_labels.npy"
        center_path = EXPERIMENT / f"artifacts/clusters/{view}_centers.npy"
        save_array(label_path, labels)
        save_array(center_path, model.cluster_centers_.astype(np.float32))
        view_results[view] = {
            "selected_k": selected_k,
            "selection_reason": selection_reason,
            "diagnostics": diagnostics,
            "clusters": cluster_summary(X, labels, model.cluster_centers_, manifest),
            "module_association": association(labels, modules),
            "labels_sha256": sha256_file(label_path),
            "centers_sha256": sha256_file(center_path),
        }

    cross_view: list[dict] = []
    pairs = [("statement", "proof"), ("statement", "joint"), ("proof", "joint")]
    for left, right in pairs:
        row = association(selected_labels[left], selected_labels[right])
        row["comparison"] = f"{left} vs {right}"
        cross_view.append(row)

    legacy: list[dict] = []
    for view, labels in selected_labels.items():
        for legacy_name, legacy_labels in (("style", style), ("domain", domain)):
            mask = legacy_labels >= 0
            row = association(labels, legacy_labels, mask)
            row["comparison"] = f"{view} vs {legacy_name}"
            legacy.append(row)

    analysis = {
        "experiment_id": config["experiment_id"],
        "views": view_results,
        "cross_view_associations": cross_view,
        "legacy_associations": legacy,
        "analysis_elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(EXPERIMENT / "artifacts/analysis.json", analysis)
    report = render_report(analysis, preparation, embedding, config)
    report_path = EXPERIMENT / "RESULTS.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "selected_k": {
                    name: payload["selected_k"] for name, payload in view_results.items()
                },
                "cross_view_associations": cross_view,
                "analysis_elapsed_seconds": analysis["analysis_elapsed_seconds"],
                "results_sha256": sha256_file(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
