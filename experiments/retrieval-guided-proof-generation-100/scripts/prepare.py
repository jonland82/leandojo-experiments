"""Freeze targets, retrieval examples, and exact prompts for the AWS pilot."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
SOURCE = REPO / CONFIG["source_embedding_experiment"]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*|[^\s]", re.UNICODE)


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


def strip_header(text: str) -> str:
    return text.split("\n", 1)[1] if "\n" in text else text


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_prompt(statement: str, examples: list[dict]) -> str:
    parts = [
        "You are completing a Lean 4 theorem in Mathlib. Return only the tactic script that goes after `by`.",
        "Do not use Markdown fences, prose, `by`, `sorry`, `admit`, `axiom`, or the target theorem itself.",
        "The script must compile at Mathlib commit 3c307701fa7e9acbdc0680d7f3b9c9fed9081740.",
    ]
    if examples:
        parts.append("Here are training theorem--proof examples:")
        for number, example in enumerate(examples, 1):
            parts.extend(
                [
                    f"EXAMPLE {number} COMPLETE SOURCE DECLARATION",
                    example["source_declaration"],
                ]
            )
    parts.extend(["TARGET STATEMENT", statement, "PROOF AFTER by"])
    return "\n\n".join(parts)


def main() -> None:
    manifests = read_jsonl(SOURCE / "inputs/manifest.jsonl")
    statements = [strip_header(row["text"]) for row in read_jsonl(SOURCE / "inputs/statement.jsonl")]
    proofs = [strip_header(row["text"]) for row in read_jsonl(SOURCE / "inputs/proof.jsonl")]
    embeddings = np.load(SOURCE / "artifacts/embeddings/statement.npy").astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    dataset_cfg = CONFIG["dataset"]
    retrieval_cfg = CONFIG["retrieval"]
    target_eligible = [
        i for i, row in enumerate(manifests)
        if row["source_split"] == dataset_cfg["target_source_split"]
        and len(statements[i]) <= dataset_cfg["maximum_target_statement_characters"]
    ]
    train = [i for i, row in enumerate(manifests) if row["source_split"] == dataset_cfg["retrieval_source_split"]]
    prompt_train = [
        i for i in train
        if len(statements[i]) + len(proofs[i]) <= retrieval_cfg["maximum_example_characters"]
    ]
    rng = np.random.RandomState(dataset_cfg["target_seed"])
    target_indices = sorted(
        int(i) for i in rng.choice(target_eligible, dataset_cfg["target_count"], replace=False)
    )

    mathlib = REPO / CONFIG["verification"]["mathlib_checkout"]
    source_cache: dict[Path, str] = {}

    def source_for(index: int) -> str:
        file_path = manifests[index]["file_path"]
        path = mathlib / file_path
        if not path.exists() and file_path.startswith("Std/"):
            path = mathlib / ".lake/packages/std" / file_path
        if not path.exists():
            raise FileNotFoundError(file_path)
        if path not in source_cache:
            source_cache[path] = path.read_text(encoding="utf-8")
        return source_cache[path]

    def source_declaration(index: int) -> str:
        source = source_for(index)
        lines = source.splitlines(keepends=True)
        start_line, start_column = manifests[index]["start"]
        end_line, end_column = manifests[index]["end"]
        absolute_start = sum(len(line) for line in lines[: start_line - 1]) + start_column - 1
        absolute_end = sum(len(line) for line in lines[: end_line - 1]) + end_column - 1
        return source[absolute_start:absolute_end]

    declarations = {i: source_declaration(i) for i in set(prompt_train) | set(target_indices)}
    prompt_train = [
        i for i in prompt_train
        if len(declarations[i]) <= retrieval_cfg["maximum_example_characters"]
    ]
    train_tokens = {i: tokenize(statements[i]) for i in prompt_train}
    document_frequency: Counter[str] = Counter()
    for tokens in train_tokens.values():
        document_frequency.update(set(tokens))
    average_length = float(np.mean([len(tokens) for tokens in train_tokens.values()]))
    n_documents = len(prompt_train)
    k1 = retrieval_cfg["bm25_k1"]
    b = retrieval_cfg["bm25_b"]

    def allowed(target: int, candidate: int, cosine: float) -> bool:
        return (
            manifests[target]["statement_sha256"] != manifests[candidate]["statement_sha256"]
            and cosine <= retrieval_cfg["maximum_duplicate_cosine"]
        )

    targets: list[dict] = []
    prompt_rows: list[dict] = []
    selection_rng = np.random.RandomState(retrieval_cfg["random_seed"])
    for target_rank, target in enumerate(target_indices):
        cosine_values = embeddings[prompt_train] @ embeddings[target]
        candidate_cosine = {candidate: float(value) for candidate, value in zip(prompt_train, cosine_values)}
        eligible = [candidate for candidate in prompt_train if allowed(target, candidate, candidate_cosine[candidate])]
        semantic = sorted(eligible, key=lambda i: (-candidate_cosine[i], i))[: retrieval_cfg["examples_per_prompt"]]

        query_counts = Counter(tokenize(statements[target]))
        bm25_scores: dict[int, float] = {}
        for candidate in eligible:
            counts = Counter(train_tokens[candidate])
            length = len(train_tokens[candidate])
            score = 0.0
            for token in query_counts:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                inverse_df = math.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * length / average_length)
                score += inverse_df * frequency * (k1 + 1.0) / denominator
            bm25_scores[candidate] = score
        bm25 = sorted(eligible, key=lambda i: (-bm25_scores[i], i))[: retrieval_cfg["examples_per_prompt"]]
        random_examples = [int(i) for i in selection_rng.choice(eligible, retrieval_cfg["examples_per_prompt"], replace=False)]
        selections = {
            "no_retrieval": [],
            "random": random_examples,
            "bm25": bm25,
            "semantic": semantic,
        }
        target_entry = {
            "target_rank": target_rank,
            "target_i": target,
            **manifests[target],
            "statement": statements[target],
            "recorded_proof": proofs[target],
            "recorded_source_declaration": declarations[target],
            "retrieval": {},
        }
        for condition in retrieval_cfg["conditions"]:
            examples = [
                {
                    "i": i,
                    "full_name": manifests[i]["full_name"],
                    "statement": statements[i],
                    "proof": proofs[i],
                    "source_declaration": declarations[i],
                    "statement_cosine": candidate_cosine[i],
                    "bm25_score": bm25_scores.get(i),
                }
                for i in selections[condition]
            ]
            prompt = build_prompt(statements[target], examples)
            prompt_bytes = len(prompt.encode("utf-8"))
            if prompt_bytes > CONFIG["generation"]["maximum_prompt_utf8_bytes"]:
                raise RuntimeError(f"prompt {target_rank}/{condition} has {prompt_bytes} UTF-8 bytes")
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            target_entry["retrieval"][condition] = [
                {key: value for key, value in example.items() if key not in {"statement", "proof", "source_declaration"}}
                for example in examples
            ]
            prompt_rows.append(
                {
                    "target_rank": target_rank,
                    "target_i": target,
                    "condition": condition,
                    "prompt": prompt,
                    "prompt_utf8_bytes": prompt_bytes,
                    "prompt_sha256": prompt_sha,
                }
            )
        targets.append(target_entry)

    inputs = EXPERIMENT / "inputs"
    write_jsonl(inputs / "targets.jsonl", targets)
    write_jsonl(inputs / "prompts.jsonl", prompt_rows)
    files = [inputs / "targets.jsonl", inputs / "prompts.jsonl", EXPERIMENT / "config.json"]
    summary = {
        "target_records_in_sample": sum(row["source_split"] == dataset_cfg["target_source_split"] for row in manifests),
        "target_records_eligible": len(target_eligible),
        "targets_selected": len(targets),
        "training_records_in_sample": len(train),
        "prompt_eligible_training_records": len(prompt_train),
        "prompts": len(prompt_rows),
        "prompt_utf8_bytes": {
            "minimum": min(row["prompt_utf8_bytes"] for row in prompt_rows),
            "mean": float(np.mean([row["prompt_utf8_bytes"] for row in prompt_rows])),
            "maximum": max(row["prompt_utf8_bytes"] for row in prompt_rows),
        },
        "files": {str(path.relative_to(EXPERIMENT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files},
    }
    write_json(EXPERIMENT / "artifacts/preparation.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
