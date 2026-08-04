"""Prepare exact, checksummed semantic inputs for the 10,000-proof sample."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG_PATH = EXPERIMENT / "config.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def normalized_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def corpus_path_matches(corpus_path: str, theorem_path: str) -> bool:
    corpus_path = normalized_path(corpus_path)
    theorem_path = normalized_path(theorem_path)
    return corpus_path == theorem_path or corpus_path.endswith("/" + theorem_path)


def distribution(values: np.ndarray) -> dict:
    return {
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "maximum": int(values.max()),
        "total": int(values.sum()),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = REPO / config["dataset"]["root"]
    random_dir = dataset / "random"

    available: list[dict] = []
    available_source: list[dict] = []
    available_counts: dict[str, int] = {}
    for split in config["dataset"]["splits"]:
        records = json.loads((random_dir / split).read_text(encoding="utf-8"))
        selected = [row for row in records if row.get("traced_tactics")]
        available_counts[split] = len(selected)
        for source_index, theorem in enumerate(selected):
            available.append(theorem)
            available_source.append({"split": split, "source_index": source_index})

    rng = np.random.RandomState(config["dataset"]["sample_seed"])
    chosen = np.sort(
        rng.choice(
            len(available),
            size=config["dataset"]["sample_size"],
            replace=False,
        )
    )
    theorems = [available[index] for index in chosen]
    sources = [available_source[index] for index in chosen]
    required_names = {theorem["full_name"] for theorem in theorems}

    corpus_candidates: dict[str, list[dict]] = defaultdict(list)
    with (dataset / "corpus.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            corpus_file = json.loads(line)
            corpus_path = corpus_file["path"]
            for premise in corpus_file["premises"]:
                if premise["full_name"] in required_names:
                    corpus_candidates[premise["full_name"]].append(
                        {"path": corpus_path, "code": premise["code"]}
                    )

    statements: list[str] = []
    ambiguous: list[dict] = []
    duplicate_names: set[str] = set()
    for theorem in theorems:
        candidates = corpus_candidates.get(theorem["full_name"], [])
        if len(candidates) > 1:
            duplicate_names.add(theorem["full_name"])
        matches = [
            candidate
            for candidate in candidates
            if corpus_path_matches(candidate["path"], theorem["file_path"])
        ]
        if len(matches) != 1:
            ambiguous.append(
                {
                    "full_name": theorem["full_name"],
                    "file_path": theorem["file_path"],
                    "candidate_paths": [candidate["path"] for candidate in candidates],
                    "matching_paths": [candidate["path"] for candidate in matches],
                }
            )
            statements.append("")
        else:
            statements.append(matches[0]["code"])
    if ambiguous:
        raise RuntimeError(
            f"statement lookup was not one-to-one for {len(ambiguous)} records: "
            + json.dumps(ambiguous[:5], ensure_ascii=False)
        )

    proofs = [
        "\n".join(step["tactic"] for step in theorem["traced_tactics"])
        for theorem in theorems
    ]
    views = {
        name: [
            template.format(statement=statement, proof=proof)
            for statement, proof in zip(statements, proofs)
        ]
        for name, template in config["views"].items()
    }

    old_artifact = REPO / "experiments/aws-10000/artifacts/proofs.json"
    old_points = json.loads(old_artifact.read_text(encoding="utf-8"))["points"]
    old_alignment = len(old_points) == len(theorems) and all(
        point["name"] == theorem["full_name"]
        and point["file"] == theorem["file_path"]
        for point, theorem in zip(old_points, theorems)
    )
    if not old_alignment:
        raise RuntimeError("the reproduced sample does not align with aws-10000/proofs.json")

    manifest: list[dict] = []
    for output_index, (available_index, theorem, source, statement, proof) in enumerate(
        zip(chosen, theorems, sources, statements, proofs)
    ):
        manifest.append(
            {
                "i": output_index,
                "available_index": int(available_index),
                "source_split": source["split"],
                "source_index": source["source_index"],
                "url": theorem["url"],
                "commit": theorem["commit"],
                "file_path": theorem["file_path"],
                "full_name": theorem["full_name"],
                "start": theorem["start"],
                "end": theorem["end"],
                "n_tactics": len(theorem["traced_tactics"]),
                "statement_sha256": sha256_bytes(statement.encode("utf-8")),
                "proof_sha256": sha256_bytes(proof.encode("utf-8")),
            }
        )

    inputs_dir = EXPERIMENT / "inputs"
    write_jsonl(inputs_dir / "manifest.jsonl", manifest)
    for name, texts in views.items():
        write_jsonl(
            inputs_dir / f"{name}.jsonl",
            [{"i": index, "text": text} for index, text in enumerate(texts)],
        )

    input_files = [inputs_dir / "manifest.jsonl"] + [
        inputs_dir / f"{name}.jsonl" for name in views
    ]
    preparation = {
        "experiment_id": config["experiment_id"],
        "sample": {
            "available_tactic_proofs": len(available),
            "available_by_split": available_counts,
            "selected": len(theorems),
            "selected_by_split": dict(Counter(row["source_split"] for row in manifest)),
            "seed": config["dataset"]["sample_seed"],
            "selected_available_indices_sha256": sha256_bytes(
                np.asarray(chosen, dtype=np.int64).tobytes()
            ),
            "aligned_with_aws_10000_artifact": old_alignment,
        },
        "statement_lookup": {
            "matched": len(statements),
            "missing_or_ambiguous": len(ambiguous),
            "selected_names_with_multiple_corpus_candidates": sorted(duplicate_names),
            "join_key": "full_name plus normalized corpus-path suffix match to file_path",
        },
        "view_sizes": {
            name: {
                "characters": distribution(
                    np.asarray([len(text) for text in texts], dtype=np.int64)
                ),
                "utf8_bytes": distribution(
                    np.asarray([len(text.encode("utf-8")) for text in texts], dtype=np.int64)
                ),
            }
            for name, texts in views.items()
        },
        "files": {
            str(path.relative_to(EXPERIMENT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_files
        },
        "source_files": {
            str(path.relative_to(REPO)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [
                CONFIG_PATH,
                *(random_dir / split for split in config["dataset"]["splits"]),
                dataset / "corpus.jsonl",
                old_artifact,
            ]
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_json(EXPERIMENT / "artifacts/preparation.json", preparation)
    print(json.dumps(preparation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
