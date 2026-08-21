"""Freeze proof-prefix and repetition-control inputs for the trajectory study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


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


def main() -> None:
    source = REPO / CONFIG["source_experiment"]
    source_manifest_path = source / "inputs/manifest.jsonl"
    source_manifest = [
        json.loads(line) for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    dataset = REPO / CONFIG["dataset"]["root"] / "random"
    available: list[dict] = []
    for split in CONFIG["dataset"]["splits"]:
        records = json.loads((dataset / split).read_text(encoding="utf-8"))
        available.extend(row for row in records if row.get("traced_tactics"))

    eligible = np.asarray(
        [
            row["i"]
            for row in source_manifest
            if row["n_tactics"] >= CONFIG["sample"]["minimum_tactics"]
        ],
        dtype=np.int64,
    )
    if len(eligible) < CONFIG["sample"]["size"]:
        raise RuntimeError(f"only {len(eligible)} eligible proofs")
    rng = np.random.RandomState(CONFIG["sample"]["seed"])
    chosen = np.sort(rng.choice(eligible, size=CONFIG["sample"]["size"], replace=False))

    rows: list[dict] = []
    for sample_i, target_i in enumerate(chosen):
        manifest = source_manifest[int(target_i)]
        theorem = available[manifest["available_index"]]
        if theorem["full_name"] != manifest["full_name"]:
            raise RuntimeError(f"dataset alignment failed at source index {target_i}")
        tactics = [step["tactic"] for step in theorem["traced_tactics"]]
        checkpoints = list(CONFIG["sample"]["checkpoints"])
        if CONFIG["sample"]["include_full_proof"] and len(tactics) not in checkpoints:
            checkpoints.append(len(tactics))
        for trajectory in ("actual", "repeat"):
            for checkpoint in checkpoints:
                selected = (
                    tactics[:checkpoint]
                    if trajectory == "actual"
                    else [tactics[0]] * checkpoint
                )
                proof = "\n".join(selected)
                rows.append(
                    {
                        "input_i": len(rows),
                        "sample_i": sample_i,
                        "target_i": int(target_i),
                        "full_name": manifest["full_name"],
                        "file_path": manifest["file_path"],
                        "trajectory": trajectory,
                        "checkpoint": checkpoint,
                        "total_tactics": len(tactics),
                        "is_full": checkpoint == len(tactics),
                        "text": "Lean 4 tactic proof:\n" + proof,
                    }
                )

    inputs_path = EXPERIMENT / "inputs/trajectories.jsonl"
    write_jsonl(inputs_path, rows)
    duplicate_pairs = sum(
        1 for row in rows if row["checkpoint"] == 1 and row["trajectory"] == "actual"
    )
    preparation = {
        "experiment_id": CONFIG["experiment_id"],
        "eligible_proofs": int(len(eligible)),
        "selected_proofs": int(len(chosen)),
        "selected_target_indices": chosen.tolist(),
        "selected_target_indices_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(),
        "trajectory_inputs": len(rows),
        "exact_duplicate_actual_repeat_pairs_at_checkpoint_1": duplicate_pairs,
        "characters": sum(len(row["text"]) for row in rows),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "inputs_sha256": sha256_file(inputs_path),
    }
    write_json(EXPERIMENT / "artifacts/preparation.json", preparation)
    print(json.dumps(preparation, indent=2))


if __name__ == "__main__":
    main()
