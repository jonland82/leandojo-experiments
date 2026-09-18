"""Create deterministic surface- and alpha-normalized statement corpora."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))

COHORTS = {
    "reference": ROOT / "experiments/semantic-embeddings-10000",
    "control": ROOT / "experiments/frontier-time-controls",
    "frontier": ROOT / "experiments/frontier-formalizations",
}
HEADER_RE = re.compile(
    r"^(?:\s*@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?:theorem|lemma)\s+(?:`[^`]+`|[^\s(:{\[]+)\s*",
    re.DOTALL,
)
TOKEN_RE = re.compile(r"(?<![\w'])([^\W\d][\w']*)(?![\w'])")
RESERVED = {
    "Type", "Prop", "Sort", "max", "outParam", "semiOutParam", "by", "where",
    "let", "in", "if", "then", "else", "match", "with", "fun", "forall",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_statement(rendered: str) -> str:
    return rendered.split("\n", 1)[1] if "\n" in rendered else rendered


def header_normalize(statement: str) -> str:
    match = HEADER_RE.match(statement.strip())
    body = statement[match.end():] if match else statement
    return " ".join(body.split())


def binder_names(text: str) -> list[str]:
    names: list[str] = []
    pairs = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}
    stack: list[tuple[str, int]] = []
    for index, character in enumerate(text):
        if character in pairs:
            stack.append((character, index))
        elif stack and character == pairs[stack[-1][0]]:
            _, start = stack.pop()
            segment = text[start + 1:index]
            colon = segment.find(":")
            if colon >= 0:
                prefix = segment[:colon]
                for token in TOKEN_RE.findall(prefix):
                    if token not in RESERVED and token != "_" and token not in names:
                        names.append(token)
    for match in re.finditer(r"(?:∀|forall)\s+([^,:=]+?)(?=\s*[:,])", text):
        for token in TOKEN_RE.findall(match.group(1)):
            if token not in RESERVED and token != "_" and token not in names:
                names.append(token)
    for match in re.finditer(r"fun\s+([^,:=↦]+?)(?=\s*(?::|=>|↦))", text):
        for token in TOKEN_RE.findall(match.group(1)):
            if token not in RESERVED and token != "_" and token not in names:
                names.append(token)
    first_positions = {}
    for match in TOKEN_RE.finditer(text):
        first_positions.setdefault(match.group(1), match.start())
    return sorted(names, key=lambda name: first_positions[name])


def alpha_normalize(statement: str) -> str:
    text = header_normalize(statement)
    mapping = {name: f"v{index}" for index, name in enumerate(binder_names(text))}
    if not mapping:
        return text
    return TOKEN_RE.sub(lambda match: mapping.get(match.group(1), match.group(1)), text)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    rows = []
    raw_texts = []
    offset = 0
    for cohort, directory in COHORTS.items():
        manifests = read_jsonl(directory / "inputs/manifest.jsonl")
        statements = read_jsonl(directory / "inputs/statement.jsonl")
        if len(manifests) != len(statements):
            raise ValueError(f"{cohort} manifest/input length mismatch")
        for manifest, statement_row in zip(manifests, statements):
            text = raw_statement(statement_row["text"])
            rows.append({
                "i": offset,
                "cohort": cohort,
                "source_i": manifest["i"],
                "project": manifest.get("project"),
                "revision": manifest.get("revision") or manifest.get("commit"),
                "name": manifest.get("name") or manifest.get("full_name"),
                "raw_sha256": digest(text),
                "raw_characters": len(text),
            })
            raw_texts.append(text)
            offset += 1

    transforms = {
        "header_normalized": header_normalize,
        "alpha_normalized": alpha_normalize,
    }
    inputs = EXPERIMENT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    with (inputs / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    view_summaries = {}
    total_characters = 0
    for variant in CONFIG["variants"]:
        transform = transforms[variant]
        changed = 0
        alpha_binders = 0
        lengths = []
        with (inputs / f"{variant}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row, raw in zip(rows, raw_texts):
                normalized = transform(raw)
                changed += normalized != raw
                if variant == "alpha_normalized":
                    alpha_binders += len(binder_names(header_normalize(raw)))
                lengths.append(len(normalized))
                rendered = CONFIG["view_template"].format(statement=normalized)
                total_characters += len(rendered)
                stream.write(json.dumps({"i": row["i"], "text": rendered,
                                         "statement_characters": len(normalized)},
                                        ensure_ascii=False) + "\n")
        view_summaries[variant] = {
            "declarations": len(rows),
            "changed": changed,
            "changed_fraction": changed / len(rows),
            "total_statement_characters": sum(lengths),
            "median_statement_characters": sorted(lengths)[len(lengths) // 2],
            "bound_identifiers_replaced": alpha_binders if variant == "alpha_normalized" else None,
        }

    embedding = CONFIG["embedding"]
    tokens = total_characters / embedding["cost_guard_characters_per_token"]
    projected = tokens / 1_000_000 * embedding["estimated_price_usd_per_million_text_tokens"]
    preparation = {
        "experiment_id": CONFIG["experiment_id"],
        "declarations": len(rows),
        "cohorts": dict(Counter(row["cohort"] for row in rows)),
        "views": view_summaries,
        "total_characters": total_characters,
        "conservative_token_estimate": tokens,
        "projected_cost_usd": projected,
        "hard_cost_cap_usd": embedding["hard_cost_cap_usd"],
        "within_cap": projected <= embedding["hard_cost_cap_usd"],
    }
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "preparation.json").write_text(
        json.dumps(preparation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(preparation, indent=2))
    if not preparation["within_cap"]:
        raise SystemExit("Projected embedding cost exceeds the hard cap; AWS was not called.")


if __name__ == "__main__":
    main()
