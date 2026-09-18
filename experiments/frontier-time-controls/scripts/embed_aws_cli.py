"""Embed time-matched statement controls with resumable AWS CLI calls."""

from __future__ import annotations

import concurrent.futures
import json
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
LOG_LOCK = threading.Lock()


def save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def log(payload: dict) -> None:
    path = EXPERIMENT / "logs/embed_batches.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK, path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload) + "\n")


def invoke(texts: list[str], start: int, temp: Path, config: dict) -> np.ndarray:
    request = {
        "texts": texts,
        "input_type": config["input_type"],
        "embedding_types": [config["embedding_type"]],
        "output_dimension": config["output_dimension"],
        "max_tokens": config["max_tokens"],
        "truncate": config["truncate"],
    }
    request_path = temp / f"{start}-request.json"
    response_path = temp / f"{start}-response.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [
        "aws", "bedrock-runtime", "invoke-model", "--region", config["region"],
        "--model-id", config["model_id"], "--content-type", "application/json",
        "--accept", "application/json", "--body", "fileb://" + request_path.resolve().as_posix(),
        str(response_path.resolve()), "--output", "json",
    ]
    for attempt in range(1, config["max_attempts"] + 1):
        response_path.unlink(missing_ok=True)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        event = {"start": start, "count": len(texts), "attempt": attempt,
                 "characters": sum(map(len, texts)), "elapsed_seconds": time.perf_counter() - started,
                 "returncode": process.returncode}
        try:
            if process.returncode:
                raise RuntimeError((process.stderr or process.stdout)[-2000:])
            response = json.loads(response_path.read_text(encoding="utf-8"))
            payload = response["embeddings"]
            if isinstance(payload, dict):
                payload = payload[config["embedding_type"]]
            array = np.asarray(payload, dtype=np.float32)
            expected = (len(texts), config["output_dimension"])
            if array.shape != expected:
                raise ValueError(f"unexpected response shape {array.shape}, expected {expected}")
            event["status"] = "ok"
            log(event)
            return array
        except Exception as error:
            event.update({"status": "error", "error": repr(error)})
            log(event)
            if attempt == config["max_attempts"]:
                raise
            time.sleep(min(60, 2 ** (attempt - 1)) + random.random())
    raise AssertionError("unreachable")


def main() -> None:
    preparation = json.loads((EXPERIMENT / "artifacts/preparation.json").read_text(encoding="utf-8"))
    config = CONFIG["embedding"]
    if not preparation["within_cap"] or preparation["projected_cost_usd"] > config["hard_cost_cap_usd"]:
        raise SystemExit("Cost guard failed; refusing to call AWS.")
    subprocess.run(
        ["aws", "sts", "get-caller-identity", "--region", config["region"]],
        check=True, capture_output=True,
    )
    texts = [json.loads(line)["text"] for line in
             (EXPERIMENT / "inputs/statement.jsonl").read_text(encoding="utf-8").splitlines()]
    chunk_dir = EXPERIMENT / "artifacts/embedding_chunks/statement"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    paths = []
    for start in range(0, len(texts), config["batch_size"]):
        end = min(len(texts), start + config["batch_size"])
        path = chunk_dir / f"{start:06d}-{end:06d}.npy"
        paths.append(path)
        valid = path.exists() and np.load(path, mmap_mode="r", allow_pickle=False).shape == (
            end - start, config["output_dimension"]
        )
        if not valid:
            pending.append((start, end, path))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="time-control-embed-") as directory:
        temp = Path(directory)
        # Four workers reduce Cohere token-rate throttling on fresh runs. All
        # chunks are resumable, so an interrupted run does not repeat successes.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(invoke, texts[start:end], start, temp, config): (start, path)
                for start, end, path in pending
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                start, path = futures[future]
                save(path, future.result())
                if completed % 20 == 0 or completed == len(futures):
                    print(f"completed {completed:,}/{len(futures):,} new batches", flush=True)
    combined = np.concatenate([np.load(path, allow_pickle=False) for path in paths])
    save(EXPERIMENT / "artifacts/embeddings/statement.npy", combined)
    run = {
        "experiment_id": CONFIG["experiment_id"],
        "model_id": config["model_id"],
        "projected_cost_usd": preparation["projected_cost_usd"],
        "hard_cost_cap_usd": config["hard_cost_cap_usd"],
        "inputs": len(texts),
        "new_calls": len(pending),
        "shape": list(combined.shape),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (EXPERIMENT / "artifacts/embedding_run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
