"""Compute dependency, semantic-connectedness, and proof-complexity scores."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import scipy
from scipy.stats import gaussian_kde, spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG_PATH = EXPERIMENT / "config.json"
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = EXPERIMENT / "figures"
SOURCE = REPO / "experiments/semantic-embeddings-10000"
NEIGHBOR_SOURCE = REPO / "experiments/semantic-neighborhood-transfer-10000"
DATASET = REPO / "data/leandojo_benchmark_4/leandojo_benchmark_4"
TACTIC_HEAD = re.compile(r"^[A-Za-z_][A-Za-z0-9_'!?]*")

COLORS = {
    "dependency": "#7857d8",
    "statement": "#158f91",
    "proof": "#dc7a38",
    "complexity": "#b83b6b",
    "ink": "#202332",
    "muted": "#697083",
    "grid": "#dfe2ea",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


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


def export_dependencies(config: dict, force: bool) -> Path:
    output = ARTIFACTS / "kernel_dependencies.tsv"
    if output.exists() and not force:
        return output

    checkout = REPO / config["runtime"]["mathlib_checkout"]
    elan_home = REPO / config["runtime"]["elan_home"]
    lake = elan_home / "bin/lake.exe"
    if not checkout.exists() or not lake.exists():
        raise FileNotFoundError(
            "Pinned Lean runtime is missing; run the retrieval experiment's "
            "scripts/setup_lean.ps1 first"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != config["dataset"]["mathlib_commit"]:
        raise RuntimeError(f"Mathlib checkout is at {commit}, not the dataset commit")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tsv.tmp")
    stderr_path = ARTIFACTS / "dependency_export_stderr.log"
    environment = os.environ.copy()
    environment["ELAN_HOME"] = str(elan_home)
    started = time.time()
    with temporary.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        subprocess.run(
            [str(lake), "env", "lean", str(EXPERIMENT / "scripts/export_dependencies.lean")],
            cwd=checkout,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    temporary.replace(output)
    print(f"Exported kernel dependencies in {time.time() - started:.1f}s", flush=True)
    return output


def load_dependency_graph(
    path: Path,
) -> tuple[list[str], list[tuple[int, ...]], dict[str, int], int]:
    raw: list[tuple[str, list[str]]] = []
    names: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if not fields[0]:
                continue
            raw.append((fields[0], fields[1:]))
            names.update(fields)
    ordered_names = sorted(names)
    ids = {name: index for index, name in enumerate(ordered_names)}
    dependencies: list[tuple[int, ...]] = [tuple() for _ in ordered_names]
    for name, dependency_names in raw:
        dependencies[ids[name]] = tuple(sorted({ids[item] for item in dependency_names}))
    return ordered_names, dependencies, ids, len(raw)


def dependency_profile(
    target: int,
    dependencies: list[tuple[int, ...]],
    maximum_distance: int,
) -> tuple[np.ndarray, bool]:
    seen = {target}
    frontier = {target}
    counts = np.zeros(maximum_distance, dtype=np.int32)
    for distance in range(maximum_distance):
        following: set[int] = set()
        for node in frontier:
            following.update(dependencies[node])
        following.difference_update(seen)
        counts[distance] = len(following)
        if not following:
            return counts, False
        seen.update(following)
        frontier = following
    return counts, bool(frontier)


def load_target_records(manifest: list[dict]) -> list[dict]:
    wanted = {(row["full_name"], row["file_path"]): row["i"] for row in manifest}
    records: list[dict | None] = [None] * len(manifest)
    for split in ("train.json", "val.json", "test.json"):
        rows = json.loads((DATASET / "random" / split).read_text(encoding="utf-8"))
        for row in rows:
            key = (row["full_name"], row["file_path"])
            if key in wanted:
                records[wanted[key]] = row
    if any(row is None for row in records):
        raise RuntimeError("Could not map every manifest row back to LeanDojo")
    return list(records)  # type: ignore[arg-type]


def tactic_entropy(record: dict) -> float:
    heads = []
    for step in record["traced_tactics"]:
        match = TACTIC_HEAD.match(step["tactic"].strip())
        heads.append(match.group(0) if match else "<anon>")
    counts = np.asarray(list(Counter(heads).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def environment_name(target: dict, name_ids: dict[str, int]) -> str | None:
    """Resolve LeanDojo's source name, including Lean-mangled private names."""
    if target["full_name"] in name_ids:
        return target["full_name"]
    module = target["file_path"]
    if module.endswith(".lean"):
        module = module[:-5]
    private_name = f"_private.{module.replace('/', '.')}.0.{target['full_name']}"
    return private_name if private_name in name_ids else None


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains a zero vector")
    return array / norms


def sample_null_cosines(vectors: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(vectors), size=size, dtype=np.int32)
    right = rng.integers(0, len(vectors), size=size, dtype=np.int32)
    same = left == right
    while np.any(same):
        right[same] = rng.integers(0, len(vectors), size=int(same.sum()), dtype=np.int32)
        same = left == right
    output = np.empty(size, dtype=np.float32)
    for start in range(0, size, 25_000):
        stop = min(start + 25_000, size)
        output[start:stop] = np.einsum(
            "ij,ij->i", vectors[left[start:stop]], vectors[right[start:stop]]
        )
    return output


def semantic_scores(top_cosines: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.maximum(top_cosines.astype(np.float64) - threshold, 0.0)
    strength = weights.sum(axis=1)
    degree = (weights > 0).sum(axis=1).astype(np.int32)
    weighted_logs = np.zeros_like(weights)
    positive = weights > 0
    weighted_logs[positive] = weights[positive] * np.log(weights[positive])
    entropy = np.zeros(len(weights), dtype=np.float64)
    nonzero = strength > 0
    entropy[nonzero] = (
        np.log(strength[nonzero])
        - weighted_logs[nonzero].sum(axis=1) / strength[nonzero]
    )
    effective = np.zeros(len(weights), dtype=np.float64)
    effective[nonzero] = np.exp(entropy[nonzero])
    return strength, effective, degree


def zscore(values: np.ndarray) -> np.ndarray:
    standard_deviation = values.std(ddof=0)
    if standard_deviation == 0:
        raise ValueError("Cannot standardize a constant component")
    return (values - values.mean()) / standard_deviation


def summary(values: np.ndarray) -> dict:
    return {
        "minimum": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(spearmanr(left, right).statistic)


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 15,
        "axes.edgecolor": COLORS["grid"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "figure.facecolor": "#f7f7fb",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#f7f7fb",
    })


def density_panel(ax: plt.Axes, values: np.ndarray, color: str, title: str, xlabel: str) -> None:
    ax.hist(values, bins=42, density=True, color=color, alpha=0.20, edgecolor="none")
    if np.unique(values).size > 2:
        grid = np.linspace(np.quantile(values, 0.002), np.quantile(values, 0.998), 350)
        ax.plot(grid, gaussian_kde(values)(grid), color=color, linewidth=2)
    ax.axvline(np.median(values), color=color, linestyle="--", linewidth=1.1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, stem: str, config: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in config["figures"]["formats"]:
        fig.savefig(
            FIGURES / f"{stem}.{extension}",
            dpi=config["figures"]["dpi"],
            bbox_inches="tight",
        )
    plt.close(fig)


def make_figures(scores: dict[str, np.ndarray], sensitivity: dict, config: dict) -> None:
    configure_plots()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
    density_panel(
        axes[0], scores["dependency_mass_log"], COLORS["dependency"],
        "Reliance", r"$\log(1+D_{0.5})$",
    )
    density_panel(
        axes[1], scores["statement_effective"], COLORS["statement"],
        "Semantic connectedness", r"effective neighbors",
    )
    density_panel(
        axes[2], scores["complexity"], COLORS["complexity"],
        "Structural complexity", r"$C(i)$",
    )
    axes[1].hist(
        scores["proof_effective"], bins=42, density=True,
        color=COLORS["proof"], alpha=0.16, edgecolor="none",
    )
    if np.unique(scores["proof_effective"]).size > 2:
        grid = np.linspace(
            np.quantile(scores["proof_effective"], 0.002),
            np.quantile(scores["proof_effective"], 0.998), 350,
        )
        axes[1].plot(
            grid, gaussian_kde(scores["proof_effective"])(grid),
            color=COLORS["proof"], linewidth=2,
        )
    axes[1].axvline(
        np.median(scores["proof_effective"]), color=COLORS["proof"],
        linestyle="--", linewidth=1.1,
    )
    axes[1].legend(
        handles=[
            Line2D([0], [0], color=COLORS["statement"], linewidth=2, label="statement"),
            Line2D([0], [0], color=COLORS["proof"], linewidth=2, label="proof"),
        ],
        frameon=False,
    )
    fig.suptitle("Three views of theorem-network position", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "metric_distributions", config)

    labels = ["dependency", "statement\nconnectedness", "proof\nconnectedness", "complexity"]
    matrix_values = np.column_stack([
        scores["dependency_mass_log"], scores["statement_effective"],
        scores["proof_effective"], scores["complexity"],
    ])
    correlation = np.asarray(spearmanr(matrix_values).statistic)
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    cmap = LinearSegmentedColormap.from_list("metric", ["#2a7490", "#ffffff", "#a13f70"])
    image = ax.imshow(correlation, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_yticks(range(4), labels)
    ax.grid(False)
    for row in range(4):
        for column in range(4):
            ax.text(column, row, f"{correlation[row, column]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.78, label="Spearman rank correlation")
    ax.set_title("The metrics capture different structure", loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "metric_correlations", config)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    plot = ax.hexbin(
        scores["statement_effective"], scores["proof_effective"], gridsize=48,
        mincnt=1, bins="log", cmap="viridis",
    )
    rho = rank_correlation(scores["statement_effective"], scores["proof_effective"])
    ax.text(
        0.03, 0.95, rf"Spearman $\rho={rho:.3f}$", transform=ax.transAxes,
        va="top", color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )
    ax.set_xlabel("effective statement neighbors")
    ax.set_ylabel("effective proof neighbors")
    ax.set_title("Connected statements need not have connected proofs", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(plot, ax=ax, label="records per hexagon (log scale)")
    fig.tight_layout()
    save_figure(fig, "statement_proof_connectedness", config)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
    dep = sensitivity["dependency_alpha"]
    axes[0].plot([row["value"] for row in dep], [row["rho"] for row in dep], marker="o", color=COLORS["dependency"])
    axes[0].set(xlabel=r"discount $\alpha$", ylabel="rank correlation", ylim=(0, 1.03))
    axes[0].set_title("Dependency discount", loc="left", fontweight="bold")
    sem = sensitivity["semantic_threshold"]
    for view, color in (("statement", COLORS["statement"]), ("proof", COLORS["proof"])):
        axes[1].plot(
            [row["quantile"] for row in sem], [row[f"{view}_rho"] for row in sem],
            marker="o", color=color, label=view,
        )
    axes[1].set(xlabel="null-pair quantile", ylim=(0, 1.03))
    axes[1].set_title("Semantic threshold", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    comp = sensitivity["complexity_leave_one_out"]
    axes[2].barh(
        [row["component"].replace("_", " ") for row in comp],
        [row["rho"] for row in comp], color=COLORS["complexity"], alpha=0.82,
    )
    axes[2].set(xlabel="rank correlation", xlim=(0, 1.03))
    axes[2].set_title("Leave-one-component-out", loc="left", fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Rankings are tested against design choices", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "rank_sensitivity", config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    dependency_path = export_dependencies(config, args.force_export)
    names, dependencies, name_ids, exported_declarations = load_dependency_graph(dependency_path)
    manifest = read_jsonl(SOURCE / "inputs/manifest.jsonl")
    records = load_target_records(manifest)
    if len(manifest) != config["dataset"]["sample_size"]:
        raise RuntimeError("Manifest size does not match frozen configuration")

    maximum_distance = config["dependency"]["maximum_distance"]
    depth_counts = np.zeros((len(manifest), maximum_distance), dtype=np.int32)
    depth_censored = np.zeros(len(manifest), dtype=bool)
    missing_targets = []
    for index, target in enumerate(manifest):
        resolved_name = environment_name(target, name_ids)
        if resolved_name is None:
            missing_targets.append(target["full_name"])
            continue
        target_id = name_ids[resolved_name]
        depth_counts[index], depth_censored[index] = dependency_profile(
            target_id, dependencies, maximum_distance
        )
        if (index + 1) % 250 == 0:
            print(f"Dependency profiles: {index + 1}/{len(manifest)}", flush=True)
    if missing_targets:
        raise RuntimeError(f"Missing {len(missing_targets)} targets from Lean environment")

    alpha = config["dependency"]["alpha"]
    distance_weights = alpha ** np.arange(1, maximum_distance + 1)
    dependency_mass = depth_counts @ distance_weights
    dependency_mass_log = np.log1p(dependency_mass)
    dependency_depth = np.where(
        depth_counts.any(axis=1),
        maximum_distance - np.argmax((depth_counts > 0)[:, ::-1], axis=1),
        0,
    ).astype(np.int32)
    direct_dependency_count = depth_counts[:, 0]

    semantic_config = config["semantic"]
    vectors = {
        view: normalize_rows(np.load(SOURCE / f"artifacts/embeddings/{view}.npy", allow_pickle=False))
        for view in ("statement", "proof")
    }
    null_cosines = {
        view: sample_null_cosines(
            array, semantic_config["null_pair_sample_size"], semantic_config["null_pair_seed"]
        )
        for view, array in vectors.items()
    }
    top_cosines = {
        view: np.load(
            NEIGHBOR_SOURCE / f"artifacts/{view}_top100_cosine.npy", allow_pickle=False
        )
        for view in ("statement", "proof")
    }
    threshold_quantile = semantic_config["threshold_quantile"]
    thresholds = {
        view: float(np.quantile(values, threshold_quantile))
        for view, values in null_cosines.items()
    }
    semantic = {
        view: semantic_scores(top_cosines[view], thresholds[view])
        for view in ("statement", "proof")
    }

    tactic_count = np.asarray([len(row["traced_tactics"]) for row in records], dtype=np.int32)
    entropy = np.asarray([tactic_entropy(row) for row in records], dtype=np.float64)
    components = {
        "log_tactic_count": zscore(np.log1p(tactic_count)),
        "log_direct_dependency_count": zscore(np.log1p(direct_dependency_count)),
        "dependency_depth": zscore(dependency_depth.astype(np.float64)),
        "tactic_head_entropy": zscore(entropy),
    }
    weights = config["complexity"]["weights"]
    complexity = sum(weights[name] * values for name, values in components.items())

    dependency_sensitivity = []
    for value in config["sensitivity"]["dependency_alpha"]:
        score = np.log1p(depth_counts @ (value ** np.arange(1, maximum_distance + 1)))
        dependency_sensitivity.append({
            "value": value,
            "rho": rank_correlation(dependency_mass_log, score),
        })
    semantic_sensitivity = []
    for quantile in config["sensitivity"]["semantic_threshold_quantiles"]:
        row = {"quantile": quantile}
        for view in ("statement", "proof"):
            threshold = float(np.quantile(null_cosines[view], quantile))
            effective = semantic_scores(top_cosines[view], threshold)[1]
            row[f"{view}_threshold"] = threshold
            row[f"{view}_rho"] = rank_correlation(semantic[view][1], effective)
        semantic_sensitivity.append(row)
    complexity_sensitivity = []
    for omitted in components:
        included = [values for name, values in components.items() if name != omitted]
        score = sum(included) / len(included)
        complexity_sensitivity.append({
            "component": omitted,
            "rho": rank_correlation(complexity, score),
        })
    sensitivity = {
        "dependency_alpha": dependency_sensitivity,
        "semantic_threshold": semantic_sensitivity,
        "complexity_leave_one_out": complexity_sensitivity,
    }

    score_arrays = {
        "dependency_mass_log": dependency_mass_log,
        "statement_effective": semantic["statement"][1],
        "proof_effective": semantic["proof"][1],
        "complexity": complexity,
    }
    matrix = np.column_stack(list(score_arrays.values()))
    correlations = np.asarray(spearmanr(matrix).statistic)
    correlation_labels = list(score_arrays)

    output_rows = []
    for index, target in enumerate(manifest):
        output_rows.append({
            "i": index,
            "full_name": target["full_name"],
            "file_path": target["file_path"],
            "dependency_mass": float(dependency_mass[index]),
            "dependency_mass_log": float(dependency_mass_log[index]),
            "direct_dependency_count": int(direct_dependency_count[index]),
            "dependency_depth": int(dependency_depth[index]),
            "dependency_depth_censored": bool(depth_censored[index]),
            "statement_weighted_degree": float(semantic["statement"][0][index]),
            "statement_effective_neighbors": float(semantic["statement"][1][index]),
            "statement_neighbor_count": int(semantic["statement"][2][index]),
            "proof_weighted_degree": float(semantic["proof"][0][index]),
            "proof_effective_neighbors": float(semantic["proof"][1][index]),
            "proof_neighbor_count": int(semantic["proof"][2][index]),
            "tactic_count": int(tactic_count[index]),
            "tactic_head_entropy": float(entropy[index]),
            "structural_complexity": float(complexity[index]),
        })
    write_jsonl(ARTIFACTS / "scores.jsonl", output_rows)
    with (ARTIFACTS / "dependency_depth_counts.npy").open("wb") as stream:
        np.save(stream, depth_counts, allow_pickle=False)

    exemplars = {}
    for label, values in score_arrays.items():
        ordering = np.argsort(values)
        selected = list(ordering[:3]) + list(ordering[-3:][::-1])
        exemplars[label] = [output_rows[int(index)] for index in selected]

    analysis = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "n_targets": len(manifest),
        "dependency_graph": {
            "exported_declarations": exported_declarations,
            "graph_nodes": len(names),
            "edges": int(sum(map(len, dependencies))),
            "maximum_distance": maximum_distance,
            "censored_targets": int(depth_censored.sum()),
            "alpha": alpha,
            "direct_dependency_count": summary(direct_dependency_count),
            "dependency_depth": summary(dependency_depth),
            "dependency_mass_log": summary(dependency_mass_log),
        },
        "semantic": {
            "model": "cohere.embed-v4:0",
            "dimension": int(vectors["statement"].shape[1]),
            "threshold_rule": f"fixed-pair null cosine quantile {threshold_quantile}",
            "top_neighbors_considered": int(top_cosines["statement"].shape[1]),
            "statement": {
                "threshold": thresholds["statement"],
                "weighted_degree": summary(semantic["statement"][0]),
                "effective_neighbors": summary(semantic["statement"][1]),
                "neighbor_count": summary(semantic["statement"][2]),
            },
            "proof": {
                "threshold": thresholds["proof"],
                "weighted_degree": summary(semantic["proof"][0]),
                "effective_neighbors": summary(semantic["proof"][1]),
                "neighbor_count": summary(semantic["proof"][2]),
            },
            "statement_proof_effective_neighbor_spearman": rank_correlation(
                semantic["statement"][1], semantic["proof"][1]
            ),
        },
        "complexity": {
            "weights": weights,
            "tactic_count": summary(tactic_count),
            "direct_dependency_count": summary(direct_dependency_count),
            "dependency_depth": summary(dependency_depth),
            "tactic_head_entropy": summary(entropy),
            "score": summary(complexity),
        },
        "correlations": {
            "labels": correlation_labels,
            "spearman": correlations.tolist(),
        },
        "sensitivity": sensitivity,
        "exemplars": exemplars,
        "provenance": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "dependency_export_sha256": sha256_file(dependency_path),
            "manifest_sha256": sha256_file(SOURCE / "inputs/manifest.jsonl"),
            "runtime_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    write_json(ARTIFACTS / "analysis.json", analysis)
    make_figures(score_arrays, sensitivity, config)
    print(json.dumps({
        "analysis": str(ARTIFACTS / "analysis.json"),
        "runtime_seconds": analysis["provenance"]["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
