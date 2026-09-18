"""Embed prepared inputs with a mandatory preflight cost guard."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_inputs(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as stream:
        for expected, line in enumerate(stream):
            row = json.loads(line)
            if row["i"] != expected:
                raise ValueError(f"nonsequential input at {expected}")
            texts.append(row["text"])
    return texts


def invoke(texts: list[str], view: str, start: int, temp: Path, config: dict) -> np.ndarray:
    request = {
        "texts": texts,
        "input_type": config["input_type"],
        "embedding_types": [config["embedding_type"]],
        "output_dimension": config["output_dimension"],
        "max_tokens": config["max_tokens"],
        "truncate": config["truncate"],
    }
    request_path = temp / f"{view}-{start}-request.json"
    response_path = temp / f"{view}-{start}-response.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [
        "aws", "bedrock-runtime", "invoke-model", "--region", config["region"],
        "--model-id", config["model_id"], "--content-type", "application/json",
        "--accept", "application/json", "--body", "fileb://" + request_path.resolve().as_posix(),
        str(response_path.resolve()), "--output", "json",
    ]
    log = EXPERIMENT / "logs/embed_batches.jsonl"
    for attempt in range(1, config["max_attempts"] + 1):
        response_path.unlink(missing_ok=True)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        event = {
            "view": view, "start": start, "count": len(texts), "attempt": attempt,
            "characters": sum(map(len, texts)), "elapsed_seconds": time.perf_counter() - started,
            "returncode": process.returncode,
        }
        try:
            if process.returncode != 0:
                raise RuntimeError((process.stderr or process.stdout)[-2000:])
            response = json.loads(response_path.read_text(encoding="utf-8"))
            payload = response["embeddings"]
            if isinstance(payload, dict):
                payload = payload[config["embedding_type"]]
            array = np.asarray(payload, dtype=np.float32)
            if array.shape != (len(texts), config["output_dimension"]):
                raise ValueError(f"unexpected response shape {array.shape}")
            event["status"] = "ok"
            append_jsonl(log, event)
            return array
        except Exception as error:
            event.update({"status": "error", "error": repr(error)})
            append_jsonl(log, event)
            if attempt == config["max_attempts"]:
                raise
            time.sleep(min(60, 2 ** (attempt - 1)) + random.random())
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", choices=tuple(CONFIG["views"]), default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    preparation = json.loads((EXPERIMENT / "artifacts/preparation.json").read_text(encoding="utf-8"))
    if not preparation["within_cap"] or preparation["projected_cost_usd"] > CONFIG["embedding"]["hard_cost_cap_usd"]:
        raise SystemExit("Cost guard failed; refusing to call AWS.")

    config = CONFIG["embedding"]
    subprocess.run(["aws", "sts", "get-caller-identity", "--region", config["region"]], check=True, capture_output=True)
    views = args.views or list(CONFIG["views"])
    summaries = {}
    run_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="frontier-embed-") as directory:
        temp = Path(directory)
        for view in views:
            texts = load_inputs(EXPERIMENT / f"inputs/{view}.jsonl")
            chunk_dir = EXPERIMENT / f"artifacts/embedding_chunks/{view}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            new_calls = 0
            pending = []
            for start in range(0, len(texts), config["batch_size"]):
                end = min(len(texts), start + config["batch_size"])
                path = chunk_dir / f"{start:06d}-{end:06d}.npy"
                paths.append(path)
                valid = False
                if path.exists():
                    prior = np.load(path, mmap_mode="r", allow_pickle=False)
                    valid = prior.shape == (end - start, config["output_dimension"])
                    del prior
                if not valid:
                    pending.append((start, end, path))
            completed_batches = len(paths) - len(pending)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(invoke, texts[start:end], view, start, temp, config): (start, end, path)
                    for start, end, path in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    start, end, path = futures[future]
                    save_array(path, future.result())
                    new_calls += 1
                    completed_batches += 1
                    if completed_batches % 20 == 0 or completed_batches == len(paths):
                        embedded = min(len(texts), completed_batches * config["batch_size"])
                        print(f"{view}: {embedded:,}/{len(texts):,}", flush=True)
            combined = np.concatenate([np.load(path, allow_pickle=False) for path in paths])
            output = EXPERIMENT / f"artifacts/embeddings/{view}.npy"
            save_array(output, combined)
            summaries[view] = {"inputs": len(texts), "new_calls": new_calls, "shape": list(combined.shape)}

    run = {
        "experiment_id": CONFIG["experiment_id"],
        "projected_cost_usd": preparation["projected_cost_usd"],
        "hard_cost_cap_usd": config["hard_cost_cap_usd"],
        "views": summaries,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    path = EXPERIMENT / "artifacts/embedding_run.json"
    path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
