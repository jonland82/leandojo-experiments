"""Extract comparable static architecture features from selected declarations."""

from __future__ import annotations

import json
import re
import subprocess
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
TIME = ROOT / "experiments/frontier-time-controls"
FRONTIER = ROOT / "experiments/frontier-formalizations"
TIME_CONFIG = json.loads((TIME / "config.json").read_text(encoding="utf-8"))
MATHLIB = ROOT / "data/mathlib4-history"
FRONTIER_SOURCES = ROOT / "data/frontier_sources"

TARGET_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*?\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?P<kind>theorem|lemma)\s+(?P<name>[^\s(:{\[]+)"
)
DECL_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*?\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?P<kind>theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|axiom|opaque|example)"
    r"(?:\s+(?P<name>[^\s(:{\[]+))?"
)
TOKEN_RE = re.compile(r"(?<![\w'])([^\W\d][\w']*)(?![\w'])")
IMPORT_RE = re.compile(r"(?m)^\s*import\s+")
SCOPE_RE = re.compile(r"^\s*(namespace|section|end)\b")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def split_declaration(block: str, search_start: int) -> tuple[str, str] | None:
    marker = block.find(":=", search_start)
    width = 2
    if marker < 0:
        match = re.search(r"(?m)^\s*by\s*$", block[search_start:])
        if not match:
            return None
        marker = search_start + match.start()
        width = len(match.group(0))
    statement = block[:marker].strip()
    proof = block[marker + width:].strip()
    return (statement, proof) if statement and proof else None


def maximum_depth(text: str) -> int:
    pairs = {")": "(", "}": "{", "]": "[", "⦄": "⦃"}
    opens = set(pairs.values())
    stack = []
    maximum = 0
    for character in text:
        if character in opens:
            stack.append(character)
            maximum = max(maximum, len(stack))
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return maximum


def binder_count(pattern: str, statement: str) -> int:
    total = 0
    for match in re.finditer(pattern, statement, re.DOTALL):
        prefix = match.group(1).split(":", 1)[0]
        total += sum(token != "_" for token in TOKEN_RE.findall(prefix))
    return total


def scope_depths(lines: list[str]) -> list[int]:
    """Return the lexical scope depth before each one-based source line."""
    depths = [0] * (len(lines) + 1)
    depth = 0
    for line_number, line in enumerate(lines, 1):
        depths[line_number] = depth
        match = SCOPE_RE.match(line)
        if not match:
            continue
        if match.group(1) in {"namespace", "section"}:
            depth += 1
        else:
            depth = max(0, depth - 1)
    return depths


def scan_file(path: Path, repo: Path, requested: list[dict], cohort: str) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    newline_positions = [match.start() for match in re.finditer("\n", text)]
    depth_by_line = scope_depths(lines)
    declarations = list(DECL_RE.finditer(text))
    starts = [match.start() for match in declarations]
    target_by_line: dict[int, list[dict]] = defaultdict(list)
    target_by_name: dict[str, list[dict]] = defaultdict(list)
    for row in requested:
        target_by_line[row["line"]].append(row)
        target_by_name[row["name"]].append(row)
    matched_source_ids: set[int] = set()
    targets = []
    prior_names: list[str] = []
    prior_definitions = 0
    prior_total = 0
    for match in declarations:
        line = bisect_right(newline_positions, match.start()) + 1
        kind = match.group("kind")
        name = (match.group("name") or "").strip("`")
        row = next(
            (candidate for candidate in target_by_line.get(line, [])
             if candidate["i"] not in matched_source_ids),
            None,
        )
        if row is None and name:
            row = next(
                (candidate for candidate in target_by_name.get(name, [])
                 if candidate["i"] not in matched_source_ids
                 and abs(candidate["line"] - line) <= 5),
                None,
            )
        if row is not None and kind in {"theorem", "lemma"}:
            position = bisect_right(starts, match.start())
            end = starts[position] if position < len(starts) else len(text)
            block = text[match.start():end]
            parts = split_declaration(block, match.end() - match.start())
            if parts is None:
                raise RuntimeError(f"could not split selected declaration {path}:{line}")
            statement, proof = parts
            proof_tokens = TOKEN_RE.findall(proof)
            prior_set = set(prior_names)
            local_hits = {token for token in proof_tokens if token in prior_set}
            targets.append({
                "cohort": cohort,
                "source_i": row["i"],
                "project": row.get("project"),
                "revision": row.get("revision") or row.get("commit"),
                "file_path": row["file_path"],
                "line": line,
                "name": row.get("name", name),
                "statement_characters": len(statement),
                "proof_characters": len(proof),
                "file_lines": len(lines),
                "file_declarations": len(declarations),
                "relative_declaration_position": line / max(1, len(lines)),
                "direct_imports": len(IMPORT_RE.findall(text)),
                "scope_depth": depth_by_line[min(line, len(lines))],
                "explicit_binders": binder_count(r"\(([^()]*(?::)[^()]*)\)", statement),
                "implicit_binders": binder_count(r"\{([^{}]*(?::)[^{}]*)\}", statement),
                "instance_binders": len(re.findall(r"\[[^\[\]]+\]", statement)),
                "arrow_forall_count": statement.count("→") + statement.count("∀")
                + len(re.findall(r"\bforall\b|->", statement)),
                "delimiter_depth": maximum_depth(statement),
                "prior_declarations": prior_total,
                "prior_definition_fraction": prior_definitions / max(1, prior_total),
                "local_references": len(local_hits),
                "proof_unique_tokens": len(set(proof_tokens)),
                "local_reference_fraction": len(local_hits) / max(1, len(set(proof_tokens))),
            })
            matched_source_ids.add(row["i"])
        if name:
            prior_names.append(name.split(".")[-1])
        prior_total += 1
        if kind not in {"theorem", "lemma", "example"}:
            prior_definitions += 1
    missing = sorted(row["line"] for row in requested if row["i"] not in matched_source_ids)
    if missing:
        raise RuntimeError(f"missing {len(missing)} selected declarations in {path}: {missing[:5]}")
    return targets


def extract_group(repo: Path, rows: list[dict], cohort: str) -> list[dict]:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["file_path"]].append(row)
    output = []
    for number, (relative, selected) in enumerate(sorted(by_file.items()), 1):
        path = repo / relative
        if not path.exists():
            raise FileNotFoundError(path)
        output.extend(scan_file(path, repo, selected, cohort))
        if number % 500 == 0:
            print(f"{cohort}: scanned {number:,}/{len(by_file):,} selected files", flush=True)
    return output


def main() -> None:
    controls = read_jsonl(TIME / "inputs/manifest.jsonl")
    frontier = read_jsonl(FRONTIER / "inputs/manifest.jsonl")
    features = []
    by_revision: dict[str, list[dict]] = defaultdict(list)
    for row in controls:
        by_revision[row["revision"]].append(row)
    for revision, rows in sorted(by_revision.items()):
        git(MATHLIB, "checkout", "--detach", "--force", revision)
        extracted = extract_group(MATHLIB, rows, "control")
        features.extend(extracted)
        print(f"control {revision[:8]}: extracted {len(extracted):,}", flush=True)

    by_project: dict[str, list[dict]] = defaultdict(list)
    for row in frontier:
        if row["project"] in TIME_CONFIG["projects"]:
            by_project[row["project"]].append(row)
    for project, rows in sorted(by_project.items()):
        extracted = extract_group(FRONTIER_SOURCES / project, rows, "frontier")
        features.extend(extracted)
        print(f"frontier {project}: extracted {len(extracted):,}", flush=True)

    features.sort(key=lambda row: (0 if row["cohort"] == "control" else 1, row["source_i"]))
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "features.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in features:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "experiment_id": "frontier-architecture-controls",
        "control_declarations": sum(row["cohort"] == "control" for row in features),
        "frontier_declarations": sum(row["cohort"] == "frontier" for row in features),
        "projects": {project: len(rows) for project, rows in sorted(by_project.items())},
        "feature_file": "artifacts/features.jsonl",
    }
    (artifacts / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
