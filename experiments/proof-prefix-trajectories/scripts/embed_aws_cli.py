"""Embed trajectory inputs with resumable Cohere Embed v4 AWS CLI calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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


def invoke(texts: list[str], start: int, temporary: Path) -> tuple[np.ndarray, int, float]:
    config = CONFIG["embedding"]
    request = {
        "texts": texts,
        "input_type": config["input_type"],
        "embedding_types": [config["embedding_type"]],
        "output_dimension": config["output_dimension"],
        "max_tokens": config["max_tokens"],
        "truncate": config["truncate"],
    }
    request_path = temporary / f"request-{start:05d}.json"
    response_path = temporary / f"response-{start:05d}.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    command = [
        "aws", "bedrock-runtime", "invoke-model",
        "--region", config["region"],
        "--model-id", config["model_id"],
        "--content-type", "application/json",
        "--accept", "application/json",
        "--body", "fileb://" + request_path.resolve().as_posix(),
        str(response_path.resolve()),
        "--output", "json",
    ]
    for attempt in range(1, config["max_attempts"] + 1):
        response_path.unlink(missing_ok=True)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if process.returncode == 0 and response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            payload = response["embeddings"]
            if isinstance(payload, dict):
                payload = payload[config["embedding_type"]]
            array = np.asarray(payload, dtype=np.float32)
            if array.shape == (len(texts), config["output_dimension"]) and np.isfinite(array).all():
                return array, attempt, elapsed
        if attempt == config["max_attempts"]:
            error = (process.stderr or process.stdout)[-2000:]
            raise RuntimeError(f"embedding batch {start} failed: {error}")
        time.sleep(min(60, 2 ** (attempt - 1)) + random.random())
    raise AssertionError("unreachable")


def main() -> None:
    rows = [
        json.loads(line)
        for line in (EXPERIMENT / "inputs/trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    texts = [row["text"] for row in rows]
    config = CONFIG["embedding"]
    chunks = EXPERIMENT / "artifacts/embedding_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    calls = reused = retries = 0
    elapsed_total = 0.0
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="proof-prefix-bedrock-") as temporary_name:
        temporary = Path(temporary_name)
        for start in range(0, len(texts), config["batch_size"]):
            end = min(len(texts), start + config["batch_size"])
            path = chunks / f"{start:05d}-{end:05d}.npy"
            paths.append(path)
            if path.exists():
                existing = np.load(path, allow_pickle=False, mmap_mode="r")
                valid = existing.shape == (end - start, config["output_dimension"])
                del existing
                if valid:
                    reused += 1
                    continue
            array, attempts, elapsed = invoke(texts[start:end], start, temporary)
            save_array(path, array)
            calls += 1
            retries += attempts - 1
            elapsed_total += elapsed
            print(f"embedded {end:,}/{len(texts):,}", flush=True)

    combined = np.concatenate([np.load(path, allow_pickle=False) for path in paths])
    output = EXPERIMENT / "artifacts/embeddings.npy"
    save_array(output, combined.astype(np.float32, copy=False))
    summary_path = EXPERIMENT / "artifacts/embedding_run.json"
    previous = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    previous_calls = previous.get(
        "cumulative_new_aws_cli_calls", previous.get("new_aws_cli_calls_this_run", 0)
    )
    previous_retries = previous.get(
        "cumulative_retries", previous.get("retries_this_run", 0)
    )
    previous_seconds = previous.get(
        "cumulative_invoke_seconds", previous.get("invoke_seconds_this_run", 0.0)
    )
    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "inputs": len(texts),
        "characters": sum(map(len, texts)),
        "shape": list(combined.shape),
        "new_aws_cli_calls_this_run": calls,
        "reused_batches_this_run": reused,
        "retries_this_run": retries,
        "invoke_seconds_this_run": elapsed_total,
        "cumulative_new_aws_cli_calls": previous_calls + calls,
        "cumulative_retries": previous_retries + retries,
        "cumulative_invoke_seconds": previous_seconds + elapsed_total,
        "sha256": sha256_file(output),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
