"""Anonymize conservatively resolved project-defined global identifiers."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
STYLE = ROOT / "experiments/frontier-style-ablation"
GRAPH = ROOT / "experiments/frontier-dependency-shortcuts/artifacts/graph_nodes.jsonl"
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
IDENT_RE = re.compile(r"(?<![\w'])([^\W\d][\w']*(?:\.[^\W\d][\w']*)*)(?![\w'])")
PLACEHOLDER_RE = re.compile(r"^[vg]\d+$")


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def safe_unqualified(name: str) -> bool:
    """Reject names easily confused with auto-implicit or local identifiers."""
    if PLACEHOLDER_RE.match(name) or len(name) < 3:
        return False
    if name[0].isupper():
        return True
    if "_" in name or any(character.isdigit() for character in name):
        return True
    return False


def main() -> None:
    manifest = jsonl(STYLE / "inputs/manifest.jsonl")
    alpha = jsonl(STYLE / "inputs/alpha_normalized.jsonl")
    frontier = [(row, text) for row, text in zip(manifest, alpha) if row["cohort"] == "frontier"]
    token_pool: dict[str, set[str]] = defaultdict(set)
    for row, text in frontier:
        token_pool[row["project"]].update(IDENT_RE.findall(text["text"].split("\n", 1)[-1]))

    exact: dict[str, set[str]] = defaultdict(set)
    short_counts: dict[str, Counter] = defaultdict(Counter)
    with GRAPH.open(encoding="utf-8") as stream:
        for line in stream:
            node = json.loads(line)
            project = node["project"]
            name = node["name"]
            full = node["full_name"]
            short = name.split(".")[-1]
            if name in token_pool[project] and ("." in name or safe_unqualified(name)):
                exact[project].add(name)
            if full in token_pool[project] and ("." in full or safe_unqualified(full)):
                exact[project].add(full)
            if short in token_pool[project] and safe_unqualified(short):
                short_counts[project][short] += 1
    unique_short = {project: {name for name, count in counts.items() if count == 1}
                    for project, counts in short_counts.items()}

    inputs = EXPERIMENT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    project_summary = defaultdict(lambda: {"declarations": 0, "changed": 0, "replacements": 0,
                                           "distinct_resolved_symbols": set()})
    total_characters = 0
    with (inputs / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest_stream, \
         (inputs / "vocabulary_anonymized.jsonl").open("w", encoding="utf-8", newline="\n") as text_stream:
        for output_i, (row, text_row) in enumerate(frontier):
            project = row["project"]
            rendered = text_row["text"]
            prefix, statement = rendered.split("\n", 1)
            mapping: dict[str, str] = {}
            replacements = 0

            def replace(match: re.Match) -> str:
                nonlocal replacements
                token = match.group(1)
                if token not in exact[project] and not ("." not in token and token in unique_short[project]):
                    return token
                if token not in mapping:
                    mapping[token] = f"g{len(mapping)}"
                replacements += 1
                return mapping[token]

            transformed = IDENT_RE.sub(replace, statement)
            output = prefix + "\n" + transformed
            summary = project_summary[project]
            summary["declarations"] += 1
            summary["changed"] += transformed != statement
            summary["replacements"] += replacements
            summary["distinct_resolved_symbols"].update(mapping)
            total_characters += len(output)
            manifest_stream.write(json.dumps({"i": output_i, "source_i": row["source_i"],
                                               "project": project, "name": row["name"]}) + "\n")
            text_stream.write(json.dumps({"i": output_i, "text": output,
                                          "statement_characters": len(transformed)},
                                         ensure_ascii=False) + "\n")
    embedding = CONFIG["embedding"]
    tokens = total_characters / embedding["cost_guard_characters_per_token"]
    projected = tokens / 1_000_000 * embedding["estimated_price_usd_per_million_text_tokens"]
    projects = {project: {**summary, "distinct_resolved_symbols": len(summary["distinct_resolved_symbols"]),
                          "changed_fraction": summary["changed"] / summary["declarations"]}
                for project, summary in sorted(project_summary.items())}
    result = {"experiment_id": CONFIG["experiment_id"], "declarations": len(frontier),
              "changed": sum(value["changed"] for value in projects.values()),
              "replacements": sum(value["replacements"] for value in projects.values()),
              "total_characters": total_characters, "projects": projects,
              "conservative_token_estimate": tokens, "projected_cost_usd": projected,
              "hard_cost_cap_usd": embedding["hard_cost_cap_usd"],
              "within_cap": projected <= embedding["hard_cost_cap_usd"]}
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "preparation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["within_cap"]:
        raise SystemExit("Projected embedding cost exceeds hard cap; AWS was not called.")


if __name__ == "__main__":
    main()
