"""Analyze paired kernel-verified outcomes for the retrieval-generation pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
CONDITIONS = CONFIG["retrieval"]["conditions"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [center - radius, center + radius]


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower = min(first_only, second_only)
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_difference(first: np.ndarray, second: np.ndarray, rng: np.random.RandomState) -> list[float]:
    sample_indices = rng.randint(0, len(first), size=(10_000, len(first)))
    differences = (first[sample_indices] - second[sample_indices]).mean(axis=1)
    return [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))]


def failure_category(row: dict, response: dict) -> str:
    if row["success"]:
        return "success"
    if row.get("extraction_error"):
        return row["extraction_error"]
    if "VERIFIER_TIMEOUT" in row.get("stderr", ""):
        return "verifier_timeout"
    output = (row.get("stdout", "") + "\n" + row.get("stderr", "")).lower()
    if response.get("stop_reason") == "max_tokens":
        return "output_cap_truncated"
    if "unknown identifier" in output or "unknown constant" in output:
        return "unknown_identifier"
    if "unsolved goals" in output:
        return "unsolved_goals"
    if any(token in output for token in ["unexpected token", "expected token", "unexpected end", "unterminated"]):
        return "syntax_or_parse_error"
    if "tactic" in output and "failed" in output:
        return "tactic_failure"
    if any(token in output for token in ["type mismatch", "failed to synthesize", "application type mismatch", "invalid field"]):
        return "elaboration_or_type_error"
    return "other_lean_rejection"


def main() -> None:
    targets = read_jsonl(EXPERIMENT / "inputs/targets.jsonl")
    prompts = read_jsonl(EXPERIMENT / "inputs/prompts.jsonl")
    responses = read_jsonl(EXPERIMENT / "outputs/responses.jsonl")
    attempts = read_jsonl(EXPERIMENT / "verification/attempts.jsonl")
    generation_run = json.loads((EXPERIMENT / "artifacts/generation_run.json").read_text(encoding="utf-8"))
    expected_keys = {
        (target, condition, candidate)
        for target in range(len(targets))
        for condition in CONDITIONS
        for candidate in range(CONFIG["generation"]["candidates_per_condition"])
    }
    response_by_key = {(row["target_rank"], row["condition"], row["candidate_index"]): row for row in responses}
    attempt_by_key = {(row["target_rank"], row["condition"], row["candidate_index"]): row for row in attempts}
    if set(response_by_key) != expected_keys or set(attempt_by_key) != expected_keys:
        raise RuntimeError("response or verification keys are incomplete or duplicated")
    if any(attempt_by_key[key].get("response_sha256") != response_by_key[key]["response_sha256"] for key in expected_keys):
        raise RuntimeError("verification responses do not align with generation responses")

    success = {
        condition: np.asarray(
            [
                [bool(attempt_by_key[(target, condition, candidate)]["success"]) for candidate in range(3)]
                for target in range(len(targets))
            ],
            dtype=bool,
        )
        for condition in CONDITIONS
    }
    condition_metrics: dict[str, dict] = {}
    for condition in CONDITIONS:
        condition_responses = [row for row in responses if row["condition"] == condition]
        condition_attempts = [row for row in attempts if row["condition"] == condition]
        pass_at_k = {}
        for k in range(1, 4):
            per_target = success[condition][:, :k].any(axis=1)
            count = int(per_target.sum())
            pass_at_k[str(k)] = {
                "successes": count,
                "targets": len(targets),
                "rate": count / len(targets),
                "wilson_95": wilson(count, len(targets)),
            }
        condition_metrics[condition] = {
            "accepted_attempts": sum(row["success"] for row in condition_attempts),
            "attempts": len(condition_attempts),
            "attempt_success_rate": sum(row["success"] for row in condition_attempts) / len(condition_attempts),
            "pass_at_k": pass_at_k,
            "input_tokens": sum(row["usage"]["inputTokens"] for row in condition_responses),
            "output_tokens": sum(row["usage"]["outputTokens"] for row in condition_responses),
            "observed_cost_usd": sum(row["cost_usd"] for row in condition_responses),
            "mean_bedrock_latency_ms": float(np.mean([row["bedrock_latency_ms"] for row in condition_responses])),
            "output_cap_truncations": sum(row["stop_reason"] == "max_tokens" for row in condition_responses),
            "generation_retries": sum(row["attempts"] - 1 for row in condition_responses),
            "mean_verification_seconds": float(np.mean([row["elapsed_seconds"] for row in condition_attempts])),
        }

    rng = np.random.RandomState(0)
    paired = {}
    semantic_pass3 = success["semantic"].any(axis=1)
    for baseline in ["no_retrieval", "random", "bm25"]:
        baseline_pass3 = success[baseline].any(axis=1)
        semantic_only = int(np.logical_and(semantic_pass3, ~baseline_pass3).sum())
        baseline_only = int(np.logical_and(~semantic_pass3, baseline_pass3).sum())
        paired["semantic_minus_" + baseline] = {
            "difference": float(semantic_pass3.mean() - baseline_pass3.mean()),
            "bootstrap_95": bootstrap_difference(semantic_pass3.astype(float), baseline_pass3.astype(float), rng),
            "both_success": int(np.logical_and(semantic_pass3, baseline_pass3).sum()),
            "semantic_only": semantic_only,
            "baseline_only": baseline_only,
            "neither_success": int(np.logical_and(~semantic_pass3, ~baseline_pass3).sum()),
            "exact_mcnemar_p_two_sided": exact_mcnemar_p(semantic_only, baseline_only),
        }

    timeout_rows = [row for row in attempts if "VERIFIER_TIMEOUT" in row.get("stderr", "")]
    timeout_targets = sorted({row["target_rank"] for row in timeout_rows})
    retained = np.asarray([rank not in timeout_targets for rank in range(len(targets))])
    timeout_sensitivity = {
        "excluded_target_ranks": timeout_targets,
        "retained_targets": int(retained.sum()),
        "pass_at_3": {
            condition: {
                "successes": int(success[condition].any(axis=1)[retained].sum()),
                "targets": int(retained.sum()),
                "rate": float(success[condition].any(axis=1)[retained].mean()),
            }
            for condition in CONDITIONS
        },
    }
    categories = Counter()
    categories_by_condition: dict[str, Counter] = {condition: Counter() for condition in CONDITIONS}
    for key, attempt in attempt_by_key.items():
        category = failure_category(attempt, response_by_key[key])
        categories[category] += 1
        categories_by_condition[attempt["condition"]][category] += 1

    prompt_by_key = {(row["target_rank"], row["condition"]): row for row in prompts}
    overlap = []
    for target in targets:
        semantic_ids = {row["i"] for row in target["retrieval"]["semantic"]}
        bm25_ids = {row["i"] for row in target["retrieval"]["bm25"]}
        overlap.append(len(semantic_ids & bm25_ids))

    files = [
        EXPERIMENT / "config.json",
        EXPERIMENT / "inputs/targets.jsonl",
        EXPERIMENT / "inputs/prompts.jsonl",
        EXPERIMENT / "outputs/responses.jsonl",
        EXPERIMENT / "verification/attempts.jsonl",
        EXPERIMENT / "artifacts/generation_run.json",
    ]
    analysis = {
        "experiment_id": CONFIG["experiment_id"],
        "integrity": {
            "targets": len(targets),
            "prompts": len(prompts),
            "responses": len(responses),
            "verification_attempts": len(attempts),
            "complete_balanced_design": True,
            "files": {
                str(path.relative_to(EXPERIMENT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in files
            },
        },
        "model": {
            "requested": CONFIG["generation"]["requested_model_id"],
            "used": CONFIG["generation"]["model_id"],
            "substitution_reason": CONFIG["generation"]["model_substitution_reason"],
        },
        "condition_metrics": condition_metrics,
        "paired_pass_at_3": paired,
        "timeout_sensitivity": timeout_sensitivity,
        "failure_categories": dict(categories),
        "failure_categories_by_condition": {key: dict(value) for key, value in categories_by_condition.items()},
        "retrieval_diagnostics": {
            "mean_semantic_bm25_example_overlap_out_of_four": float(np.mean(overlap)),
            "targets_with_no_semantic_bm25_example_overlap": int(sum(value == 0 for value in overlap)),
        },
        "generation": generation_run,
        "verification": {
            "accepted_attempts": sum(row["success"] for row in attempts),
            "timeouts": len(timeout_rows),
            "timeout_target_ranks": timeout_targets,
            "total_verifier_seconds_sum": sum(row["elapsed_seconds"] for row in attempts),
            "mean_verifier_seconds": float(np.mean([row["elapsed_seconds"] for row in attempts])),
            "maximum_verifier_seconds": max(row["elapsed_seconds"] for row in attempts),
        },
    }
    write_json(EXPERIMENT / "artifacts/analysis.json", analysis)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
