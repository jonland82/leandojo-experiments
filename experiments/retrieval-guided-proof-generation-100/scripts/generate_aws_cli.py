"""Run the budget-guarded Bedrock pilot exclusively through the AWS CLI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
import os


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cost(input_tokens: int, output_tokens: int) -> float:
    budget = CONFIG["budget"]
    return (
        input_tokens * budget["input_usd_per_million_tokens"]
        + output_tokens * budget["output_usd_per_million_tokens"]
    ) / 1_000_000.0


def invoke(task: dict, request_dir: Path, aws_executable: str) -> dict:
    generation = CONFIG["generation"]
    key = f"t{task['target_rank']:03d}-{task['condition']}-c{task['candidate_index']}"
    messages_path = request_dir / f"{key}-messages.json"
    inference_path = request_dir / f"{key}-inference.json"
    messages = [{"role": "user", "content": [{"text": task["prompt"]}]}]
    inference = {
        "maxTokens": generation["maximum_output_tokens"],
        "temperature": generation["temperature"],
    }
    messages_path.write_text(json.dumps(messages, ensure_ascii=True), encoding="ascii")
    inference_path.write_text(json.dumps(inference, ensure_ascii=True), encoding="ascii")
    command = [
        aws_executable,
        "bedrock-runtime",
        "converse",
        "--region",
        generation["aws_region"],
        "--model-id",
        generation["model_id"],
        "--messages",
        f"file://{messages_path}",
        "--inference-config",
        f"file://{inference_path}",
        "--cli-read-timeout",
        "180",
        "--output",
        "json",
    ]
    errors: list[dict] = []
    environment = os.environ.copy()
    environment["AWS_CLI_FILE_ENCODING"] = "UTF-8"
    environment["AWS_CLI_OUTPUT_ENCODING"] = "UTF-8"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    for attempt in range(generation["maximum_retries"] + 1):
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=210,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            raw = json.loads(completed.stdout)
            usage = raw["usage"]
            text = "\n".join(
                block.get("text", "")
                for block in raw["output"]["message"]["content"]
                if "text" in block
            )
            raw_canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return {
                "target_rank": task["target_rank"],
                "target_i": task["target_i"],
                "condition": task["condition"],
                "candidate_index": task["candidate_index"],
                "model_id": generation["model_id"],
                "aws_region": generation["aws_region"],
                "prompt_sha256": task["prompt_sha256"],
                "prompt_utf8_bytes": task["prompt_utf8_bytes"],
                "request_sha256": hashlib.sha256((messages_path.read_bytes() + inference_path.read_bytes())).hexdigest(),
                "response_sha256": hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest(),
                "text": text,
                "stop_reason": raw.get("stopReason"),
                "usage": usage,
                "cost_usd": cost(usage["inputTokens"], usage["outputTokens"]),
                "bedrock_latency_ms": raw.get("metrics", {}).get("latencyMs"),
                "wall_seconds": elapsed,
                "attempts": attempt + 1,
                "retry_errors": errors,
                "raw_response": raw,
            }
        errors.append(
            {
                "attempt": attempt + 1,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
                "stdout": completed.stdout,
                "wall_seconds": elapsed,
            }
        )
        if (
            "ParamValidation" in completed.stderr
            or "Error parsing parameter" in completed.stderr
            or "codec can't encode" in completed.stderr
        ):
            break
        if attempt >= generation["maximum_retries"]:
            break
        time.sleep(generation["retry_base_seconds"] * (2**attempt) + random.Random(key + str(attempt)).random())
    raise RuntimeError(f"AWS CLI invocation failed after retries for {key}: {errors[-1]['stderr']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-target-rank", type=int)
    args = parser.parse_args()
    generation = CONFIG["generation"]
    budget = CONFIG["budget"]
    prompts = read_jsonl(EXPERIMENT / "inputs/prompts.jsonl")
    targets = {row["target_rank"]: row for row in read_jsonl(EXPERIMENT / "inputs/targets.jsonl")}
    responses_path = EXPERIMENT / "outputs/responses.jsonl"
    responses = read_jsonl(responses_path)
    completed_keys = {(row["target_rank"], row["condition"], row["candidate_index"]) for row in responses}
    observed_spend = budget["preflight_cost_usd"] + sum(row["cost_usd"] for row in responses)
    actual_spend = observed_spend + budget["unobserved_failed_invocation_reserve_usd"]
    prompt_input_tokens = {row["prompt_sha256"]: row["usage"]["inputTokens"] for row in responses}
    request_dir = EXPERIMENT / ".runtime/requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    aws_executable = shutil.which("aws")
    if not aws_executable:
        raise RuntimeError("aws CLI was not found")

    tasks: list[dict] = []
    for candidate_index in range(generation["candidates_per_condition"]):
        layer = []
        for prompt in prompts:
            if args.smoke_target_rank is not None and prompt["target_rank"] != args.smoke_target_rank:
                continue
            key = (prompt["target_rank"], prompt["condition"], candidate_index)
            if key in completed_keys:
                continue
            layer.append({**prompt, "candidate_index": candidate_index})
        random.Random(1000 + candidate_index).shuffle(layer)
        tasks.extend(layer)
        if args.smoke_target_rank is not None:
            break

    lock = threading.Lock()
    reserved = 0.0
    pending: dict = {}
    next_task = 0
    budget_stopped = False
    started_at = time.time()
    with ThreadPoolExecutor(max_workers=generation["concurrency"]) as pool:
        while next_task < len(tasks) or pending:
            dispatched = False
            while next_task < len(tasks) and len(pending) < generation["concurrency"]:
                task = tasks[next_task]
                known_input = prompt_input_tokens.get(task["prompt_sha256"])
                input_bound = known_input + 64 if known_input is not None else task["prompt_utf8_bytes"] + 256
                reservation = cost(input_bound, generation["maximum_output_tokens"])
                with lock:
                    fits = actual_spend + reserved + reservation <= budget["application_stop_usd"]
                    if fits:
                        reserved += reservation
                if not fits:
                    if pending:
                        break
                    budget_stopped = True
                    next_task = len(tasks)
                    break
                future = pool.submit(invoke, task, request_dir, aws_executable)
                pending[future] = (task, reservation)
                next_task += 1
                dispatched = True
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task, reservation = pending.pop(future)
                with lock:
                    reserved -= reservation
                result = future.result()
                responses.append(result)
                completed_keys.add((result["target_rank"], result["condition"], result["candidate_index"]))
                prompt_input_tokens[result["prompt_sha256"]] = result["usage"]["inputTokens"]
                actual_spend += result["cost_usd"]
                observed_spend += result["cost_usd"]
                responses.sort(key=lambda row: (row["candidate_index"], row["target_rank"], row["condition"]))
                write_jsonl(responses_path, responses)
                print(
                    f"generated {len(responses)}/1200 spend=${actual_spend:.6f} "
                    f"t={result['target_rank']} {result['condition']} c={result['candidate_index']} "
                    f"tokens={result['usage']['inputTokens']}+{result['usage']['outputTokens']}",
                    flush=True,
                )

    total_input = budget["preflight_input_tokens"] + sum(row["usage"]["inputTokens"] for row in responses)
    total_output = budget["preflight_output_tokens"] + sum(row["usage"]["outputTokens"] for row in responses)
    summary = {
        "responses": len(responses),
        "expected_responses": 100 * len(CONFIG["retrieval"]["conditions"]) * generation["candidates_per_condition"],
        "complete": len(responses) == 100 * len(CONFIG["retrieval"]["conditions"]) * generation["candidates_per_condition"],
        "budget_stopped": budget_stopped,
        "input_tokens_including_preflight": total_input,
        "output_tokens_including_preflight": total_output,
        "inference_cost_usd_including_preflight": actual_spend,
        "observed_token_cost_usd_including_preflight": observed_spend,
        "unobserved_failed_invocation_reserve_usd": budget["unobserved_failed_invocation_reserve_usd"],
        "application_stop_usd": budget["application_stop_usd"],
        "absolute_experiment_ceiling_usd": budget["absolute_experiment_ceiling_usd"],
        "elapsed_seconds_this_run": time.time() - started_at,
        "model_id": generation["model_id"],
        "aws_region": generation["aws_region"],
    }
    write_json(EXPERIMENT / "artifacts/generation_run.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
