"""Generate resumable Cohere Embed v4 vectors by invoking the AWS CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT / "config.json"


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


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def load_inputs(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for expected, line in enumerate(stream):
            row = json.loads(line)
            if row["i"] != expected:
                raise ValueError(f"nonsequential input index in {path}: {row['i']} != {expected}")
            texts.append(row["text"])
    return texts


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def parse_embeddings(response: dict, embedding_type: str) -> np.ndarray:
    payload = response["embeddings"]
    if isinstance(payload, dict):
        payload = payload[embedding_type]
    array = np.asarray(payload, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"invalid embedding response shape/content: {array.shape}")
    return array


def invoke_batch(
    texts: list[str], config: dict, temporary_dir: Path, log_path: Path, view: str, start: int
) -> tuple[np.ndarray, dict]:
    request = {
        "texts": texts,
        "input_type": config["input_type"],
        "embedding_types": [config["embedding_type"]],
        "output_dimension": config["output_dimension"],
        "max_tokens": config["max_tokens"],
        "truncate": config["truncate"],
    }
    request_path = temporary_dir / f"{view}-{start:05d}-request.json"
    response_path = temporary_dir / f"{view}-{start:05d}-response.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    command = [
        "aws",
        "bedrock-runtime",
        "invoke-model",
        "--region",
        config["region"],
        "--model-id",
        config["model_id"],
        "--content-type",
        "application/json",
        "--accept",
        "application/json",
        "--body",
        "fileb://" + request_path.resolve().as_posix(),
        str(response_path.resolve()),
        "--output",
        "json",
    ]
    for attempt in range(1, config["max_attempts"] + 1):
        if response_path.exists():
            response_path.unlink()
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        event = {
            "view": view,
            "start": start,
            "count": len(texts),
            "characters": sum(len(text) for text in texts),
            "attempt": attempt,
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
        }
        if process.returncode == 0 and response_path.exists():
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
                array = parse_embeddings(response, config["embedding_type"])
                if array.shape != (len(texts), config["output_dimension"]):
                    raise ValueError(
                        f"expected {(len(texts), config['output_dimension'])}, got {array.shape}"
                    )
                event.update({"status": "ok", "response_id": response.get("id")})
                append_jsonl(log_path, event)
                return array, event
            except Exception as error:  # response can be incomplete despite CLI success
                event.update({"status": "invalid_response", "error": repr(error)})
        else:
            error_text = (process.stderr or process.stdout).strip()
            event.update({"status": "cli_error", "error": error_text[-2000:]})
        append_jsonl(log_path, event)
        if attempt == config["max_attempts"]:
            raise RuntimeError(f"AWS CLI embedding failed after {attempt} attempts: {event}")
        delay = min(60.0, 2.0 ** (attempt - 1)) + random.random()
        time.sleep(delay)
    raise AssertionError("unreachable")


def verify_aws(region: str) -> dict:
    version = subprocess.run(
        ["aws", "--version"], capture_output=True, text=True, check=True
    )
    identity = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--region", region, "--output", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    identity_payload = json.loads(identity.stdout)
    return {
        "aws_cli_version": (version.stdout or version.stderr).strip(),
        "identity_verified": bool(identity_payload.get("Account")),
        "identity_arn_type": identity_payload.get("Arn", "").split(":")[-1].split("/")[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--views", nargs="+", choices=("statement", "proof", "joint"), default=None
    )
    args = parser.parse_args()

    full_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = full_config["embedding"]
    views = args.views or list(full_config["views"])
    aws_metadata = verify_aws(config["region"])
    artifacts = EXPERIMENT / "artifacts"
    log_path = EXPERIMENT / "logs/embed_batches.jsonl"
    chunks_root = artifacts / "embedding_chunks"
    embeddings_root = artifacts / "embeddings"
    run_started = time.perf_counter()
    summaries: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="lean-semantic-bedrock-") as temporary:
        temporary_dir = Path(temporary)
        for view in views:
            texts = load_inputs(EXPERIMENT / f"inputs/{view}.jsonl")
            view_started = time.perf_counter()
            chunk_dir = chunks_root / view
            chunk_dir.mkdir(parents=True, exist_ok=True)
            calls = 0
            reused = 0
            retries = 0
            chunk_paths: list[Path] = []
            for start in range(0, len(texts), config["batch_size"]):
                end = min(len(texts), start + config["batch_size"])
                chunk_path = chunk_dir / f"{start:05d}-{end:05d}.npy"
                chunk_paths.append(chunk_path)
                valid_existing = False
                if chunk_path.exists():
                    existing = np.load(chunk_path, allow_pickle=False, mmap_mode="r")
                    valid_existing = existing.shape == (
                        end - start,
                        config["output_dimension"],
                    ) and np.isfinite(existing).all()
                    del existing
                if valid_existing:
                    reused += 1
                    continue
                array, event = invoke_batch(
                    texts[start:end], config, temporary_dir, log_path, view, start
                )
                save_array(chunk_path, array)
                calls += 1
                retries += event["attempt"] - 1
                completed = end
                print(
                    f"{view}: {completed:,}/{len(texts):,} "
                    f"({100 * completed / len(texts):.1f}%)",
                    flush=True,
                )

            arrays = [np.load(path, allow_pickle=False) for path in chunk_paths]
            combined = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
            if combined.shape != (len(texts), config["output_dimension"]):
                raise RuntimeError(f"assembled {view} shape is {combined.shape}")
            norms = np.linalg.norm(combined, axis=1)
            output_path = embeddings_root / f"{view}.npy"
            save_array(output_path, combined)
            summaries[view] = {
                "inputs": len(texts),
                "characters": sum(len(text) for text in texts),
                "batches": len(chunk_paths),
                "new_aws_cli_calls": calls,
                "reused_batches": reused,
                "retries": retries,
                "shape": list(combined.shape),
                "dtype": str(combined.dtype),
                "norm_minimum": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_maximum": float(norms.max()),
                "elapsed_seconds": time.perf_counter() - view_started,
                "path": str(output_path.relative_to(EXPERIMENT)).replace("\\", "/"),
                "sha256": sha256_file(output_path),
            }

    total_characters = sum(summary["characters"] for summary in summaries.values())
    price = config["estimated_price_usd_per_million_text_tokens"]
    summary = {
        "experiment_id": full_config["experiment_id"],
        "transport": "AWS CLI subprocess calling bedrock-runtime invoke-model",
        "embedding_config": config,
        "aws": aws_metadata,
        "views": summaries,
        "total_characters": total_characters,
        "estimated_tokens": {
            "at_4_characters_per_token": total_characters / 4,
            "at_3_characters_per_token": total_characters / 3,
            "at_2_5_characters_per_token": total_characters / 2.5,
        },
        "estimated_embedding_cost_usd": {
            "at_4_characters_per_token": total_characters / 4 / 1_000_000 * price,
            "at_3_characters_per_token": total_characters / 3 / 1_000_000 * price,
            "at_2_5_characters_per_token": total_characters / 2.5 / 1_000_000 * price,
        },
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    write_json(artifacts / "embedding_run.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
