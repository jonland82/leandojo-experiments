"""Verify generated candidates by replacing the target declaration in its source file."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def extract_tactics(text: str) -> tuple[str, str | None]:
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:lean)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    if candidate == "by":
        candidate = ""
    elif candidate.startswith("by\n") or candidate.startswith("by\r\n"):
        candidate = candidate[2:].lstrip("\r\n")
    if not candidate:
        return candidate, "empty_candidate"
    if re.search(r"\b(sorry|admit|axiom)\b", candidate):
        return candidate, "forbidden_placeholder"
    return candidate, None


def replace_declaration(source: str, target: dict, tactics: str) -> str:
    lines = source.splitlines(keepends=True)
    start_line, start_column = target["start"]
    end_line, end_column = target["end"]
    absolute_start = sum(len(line) for line in lines[: start_line - 1]) + start_column - 1
    absolute_end = sum(len(line) for line in lines[: end_line - 1]) + end_column - 1
    search_start = max(0, absolute_start - 2000)
    statement_start = source.rfind(target["statement"], search_start, absolute_end)
    if statement_start < 0:
        raise RuntimeError(f"exact statement not found near coordinates: {target['full_name']}")
    indented = "\n".join("  " + line if line else "" for line in tactics.splitlines())
    replacement = target["statement"] + " := by\n" + indented
    return source[:statement_start] + replacement + source[absolute_end:]


def source_path_for(mathlib: Path, file_path: str) -> Path:
    direct = mathlib / file_path
    if direct.exists():
        return direct
    if file_path.startswith("Std/"):
        dependency = mathlib / ".lake/packages/std" / file_path
        if dependency.exists():
            return dependency
    raise FileNotFoundError(file_path)


def verify_one(record: dict, targets: dict[int, dict], work: Path) -> dict:
    target = targets[record["target_rank"]]
    tactics, extraction_error = extract_tactics(record["text"])
    base = {
        "target_rank": record["target_rank"],
        "target_i": target["target_i"],
        "full_name": target["full_name"],
        "condition": record["condition"],
        "candidate_index": record["candidate_index"],
        "response_sha256": record.get("response_sha256"),
        "tactics": tactics,
        "tactics_sha256": hashlib.sha256(tactics.encode("utf-8")).hexdigest(),
        "extraction_error": extraction_error,
    }
    if extraction_error:
        return {**base, "success": False, "returncode": None, "elapsed_seconds": 0.0, "stdout": "", "stderr": extraction_error}

    mathlib = REPO / CONFIG["verification"]["mathlib_checkout"]
    try:
        source_path = source_path_for(mathlib, target["file_path"])
        source = source_path.read_text(encoding="utf-8")
        rendered = replace_declaration(source, target, tactics)
    except Exception as error:
        return {
            **base,
            "success": False,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "stdout": "",
            "stderr": f"VERIFIER_RENDER_ERROR: {type(error).__name__}: {error}",
        }
    candidate_path = work / f"t{record['target_rank']:03d}-{record['condition']}-c{record['candidate_index']}.lean"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(rendered, encoding="utf-8", newline="\n")

    environment = os.environ.copy()
    elan_home = REPO / CONFIG["verification"]["elan_home"]
    environment["ELAN_HOME"] = str(elan_home)
    environment["PATH"] = str(elan_home / "bin") + os.pathsep + environment.get("PATH", "")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(elan_home / "bin/lake.exe"), "env", "lean", str(candidate_path)],
            cwd=mathlib,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CONFIG["verification"]["timeout_seconds"],
        )
        elapsed = time.perf_counter() - started
        return {
            **base,
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        return {
            **base,
            "success": False,
            "returncode": None,
            "elapsed_seconds": elapsed,
            "stdout": error.stdout or "",
            "stderr": (error.stderr or "") + "\nVERIFIER_TIMEOUT",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded-smoke", type=int, default=0)
    args = parser.parse_args()
    target_rows = read_jsonl(EXPERIMENT / "inputs/targets.jsonl")
    targets = {row["target_rank"]: row for row in target_rows}
    if args.recorded_smoke:
        ranks = np.linspace(0, len(target_rows) - 1, args.recorded_smoke, dtype=int) if args.recorded_smoke > 1 else [0]
        records = [
            {
                "target_rank": int(rank),
                "condition": "recorded_proof_smoke",
                "candidate_index": 0,
                "text": targets[int(rank)]["recorded_proof"],
                "response_sha256": None,
            }
            for rank in ranks
        ]
        output = EXPERIMENT / "verification/recorded_smoke.jsonl"
        concurrency = min(args.recorded_smoke, CONFIG["verification"]["concurrency"])
    else:
        records = read_jsonl(EXPERIMENT / "outputs/responses.jsonl")
        output = EXPERIMENT / "verification/attempts.jsonl"
        concurrency = CONFIG["verification"]["concurrency"]
        existing = read_jsonl(output) if output.exists() else []
        response_hashes = {
            (row["target_rank"], row["condition"], row["candidate_index"]): row.get("response_sha256")
            for row in records
        }
        existing = [
            row for row in existing
            if response_hashes.get((row["target_rank"], row["condition"], row["candidate_index"]))
            == row.get("response_sha256")
        ]
        completed_keys = {(row["target_rank"], row["condition"], row["candidate_index"]) for row in existing}
        records = [
            row for row in records
            if (row["target_rank"], row["condition"], row["candidate_index"]) not in completed_keys
        ]
    work = EXPERIMENT / "verification/work"
    results: list[dict] = existing if not args.recorded_smoke else []
    starting_count = len(results)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(verify_one, record, targets, work) for record in records]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"verified {starting_count + completed}/{starting_count + len(records)} success={result['success']} {result['full_name']} {result['condition']}", flush=True)
            if completed % 10 == 0:
                write_jsonl(output, sorted(results, key=lambda row: (row["target_rank"], row["condition"], row["candidate_index"])))
    results.sort(key=lambda row: (row["target_rank"], row["condition"], row["candidate_index"]))
    write_jsonl(output, results)
    successes = sum(row["success"] for row in results)
    print(json.dumps({"attempts": len(results), "successes": successes, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
