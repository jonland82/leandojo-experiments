"""Embed vocabulary-anonymized frontier statements with resumable AWS calls."""

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
LOCK = threading.Lock()


def save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def log(event: dict) -> None:
    path = EXPERIMENT / "logs/embed_batches.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event) + "\n")


def invoke(texts: list[str], start: int, temporary: Path, config: dict) -> np.ndarray:
    request = {"texts": texts, "input_type": config["input_type"],
               "embedding_types": [config["embedding_type"]],
               "output_dimension": config["output_dimension"], "max_tokens": config["max_tokens"],
               "truncate": config["truncate"]}
    request_path = temporary / f"{start}-request.json"
    response_path = temporary / f"{start}-response.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = ["aws", "bedrock-runtime", "invoke-model", "--region", config["region"],
               "--model-id", config["model_id"], "--content-type", "application/json",
               "--accept", "application/json", "--body", "fileb://" + request_path.resolve().as_posix(),
               str(response_path.resolve()), "--output", "json"]
    for attempt in range(1, config["max_attempts"] + 1):
        response_path.unlink(missing_ok=True)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        event = {"start": start, "count": len(texts), "attempt": attempt,
                 "characters": sum(map(len, texts)), "returncode": process.returncode,
                 "elapsed_seconds": time.perf_counter() - started}
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
                raise ValueError(f"unexpected shape {array.shape}, expected {expected}")
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
    if not preparation["within_cap"]:
        raise SystemExit("Cost guard failed; refusing AWS call.")
    subprocess.run(["aws", "sts", "get-caller-identity", "--region", config["region"]],
                   check=True, capture_output=True)
    with (EXPERIMENT / "inputs/vocabulary_anonymized.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    texts = [row["text"] for row in rows]
    chunks = EXPERIMENT / "artifacts/embedding_chunks/vocabulary_anonymized"
    chunks.mkdir(parents=True, exist_ok=True)
    pending, paths = [], []
    for start in range(0, len(texts), config["batch_size"]):
        end = min(len(texts), start + config["batch_size"])
        path = chunks / f"{start:06d}-{end:06d}.npy"
        paths.append(path)
        valid = path.exists() and np.load(path, mmap_mode="r", allow_pickle=False).shape == (
            end - start, config["output_dimension"])
        if not valid:
            pending.append((start, end, path))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vocabulary-ablation-") as directory:
        temporary = Path(directory)
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = {executor.submit(invoke, texts[start:end], start, temporary, config): (start, path)
                       for start, end, path in pending}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                _, path = futures[future]
                save(path, future.result())
                if completed % 25 == 0 or completed == len(futures):
                    print(f"completed {completed:,}/{len(futures):,} new batches", flush=True)
    combined = np.concatenate([np.load(path, allow_pickle=False) for path in paths])
    output = EXPERIMENT / "artifacts/embeddings/vocabulary_anonymized.npy"
    save(output, combined)
    result = {"experiment_id": CONFIG["experiment_id"], "model_id": config["model_id"],
              "inputs": len(texts), "new_calls": len(pending), "shape": list(combined.shape),
              "projected_cost_usd": preparation["projected_cost_usd"],
              "elapsed_seconds": time.perf_counter() - started}
    (EXPERIMENT / "artifacts/embedding_run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log_path = EXPERIMENT / "logs/embed_batches.jsonl"
    events = [] if not log_path.exists() else [json.loads(line) for line in log_path.read_text(
        encoding="utf-8").splitlines()]
    successful = [event for event in events if event.get("status") == "ok"]
    successful_characters = sum(event["characters"] for event in successful)
    audit = {"successful_calls": len(successful),
             "failed_or_throttled_attempts": sum(event.get("status") == "error" for event in events),
             "successful_characters": successful_characters,
             "conservative_token_estimate": successful_characters / config["cost_guard_characters_per_token"],
             "estimated_successful_output_cost_usd": successful_characters /
             config["cost_guard_characters_per_token"] / 1_000_000 *
             config["estimated_price_usd_per_million_text_tokens"]}
    (EXPERIMENT / "artifacts/spend_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
