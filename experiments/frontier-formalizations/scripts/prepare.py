"""Extract and cost a cross-project frontier-formalization corpus."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from bisect import bisect_right
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))

TARGET_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*?\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?P<kind>theorem|lemma)\s+(?P<name>[^\s(:{\[]+)"
)
BOUNDARY_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*?\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|axiom|opaque|example)\b"
)
DECLARATION_META_RE = re.compile(r"declaration:\s*[\"']?([^\"'\s]+)")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def headline_suffixes(project: dict, repo: Path) -> set[str]:
    names = set(project.get("headline_names", []))
    metadata = project.get("headline_metadata")
    if metadata and (repo / metadata).exists():
        text = (repo / metadata).read_text(encoding="utf-8", errors="replace")
        names.update(match.group(1).split(".")[-1] for match in DECLARATION_META_RE.finditer(text))
    return names


def split_declaration(block: str, search_start: int = 0) -> tuple[str, str] | None:
    """Split after the declaration header, not inside an attribute option."""
    marker = block.find(":=", search_start)
    width = 2
    if marker < 0:
        match = re.search(r"(?m)^\s*by\s*$", block[search_start:])
        if not match:
            return None
        marker, width = search_start + match.start(), len(match.group(0))
    statement = block[:marker].strip()
    proof = block[marker + width :].strip()
    if not statement or not proof:
        return None
    return statement, proof


def extract_file(
    path: Path, repo: Path, project: dict, headlines: set[str], commit: str
) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    boundaries = list(BOUNDARY_RE.finditer(text))
    boundary_starts = [match.start() for match in boundaries]
    rows: list[dict] = []
    max_chars = CONFIG["maximum_declaration_characters"]
    for match in TARGET_RE.finditer(text):
        position = bisect_right(boundary_starts, match.start())
        end = boundary_starts[position] if position < len(boundary_starts) else len(text)
        block = text[match.start() : end]
        parts = split_declaration(block, match.end() - match.start())
        if parts is None:
            continue
        statement, proof = parts
        original_chars = len(statement) + len(proof)
        truncated = original_chars > max_chars
        if truncated:
            allowance = max(1, max_chars - len(statement))
            proof = proof[:allowance]
        relative = path.relative_to(repo).as_posix()
        name = match.group("name").strip("`")
        rows.append(
            {
                "project": project["id"],
                "cohort": project["cohort"],
                "repository": project["repository"],
                "commit": commit,
                "file_path": relative,
                "line": text.count("\n", 0, match.start()) + 1,
                "kind": match.group("kind"),
                "name": name,
                "headline": name in headlines,
                "statement": statement,
                "proof": proof,
                "statement_sha256": sha256(statement),
                "proof_sha256": sha256(proof),
                "statement_characters": len(statement),
                "proof_characters": len(proof),
                "source_characters": original_chars,
                "truncated": truncated,
            }
        )
    return rows


def main() -> None:
    source_root = ROOT / CONFIG["source_root"]
    all_rows: list[dict] = []
    project_summary: dict[str, dict] = {}
    rng = random.Random(CONFIG["sample_seed"])
    for project in CONFIG["projects"]:
        repo = source_root / project["id"]
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"missing frozen source checkout: {repo}")
        excluded = tuple(project.get("exclude_prefixes", []))
        headlines = headline_suffixes(project, repo)
        commit = git(repo, "rev-parse", "HEAD")
        rows: list[dict] = []
        files = [
            path
            for path in sorted(repo.rglob("*.lean"))
            if ".git" not in path.parts
            and ".lake" not in path.parts
            and not path.relative_to(repo).as_posix().startswith(excluded)
        ]
        total_files = len(files)
        if project["id"] == "flt":
            strata = {
                "Theorems/": 2500,
                "P2M/Sol/": 2500,
                "Definitions/": 250,
            }
            sampled_files: list[Path] = []
            for prefix, limit in strata.items():
                candidates = [
                    path for path in files
                    if path.relative_to(repo).as_posix().startswith(prefix)
                ]
                rng.shuffle(candidates)
                sampled_files.extend(candidates[:limit])
            mandatory_files = [repo / "FinalCheck.lean", repo / "Theorems/Thm_fermat_last_theorem.lean"]
            files = sorted(set(sampled_files + [path for path in mandatory_files if path.exists()]))
        for number, path in enumerate(files, 1):
            rows.extend(extract_file(path, repo, project, headlines, commit))
            if number % 1000 == 0:
                print(f"{project['id']}: scanned {number:,}/{len(files):,} files", flush=True)
        extracted = len(rows)
        limit = CONFIG["flt_sample_size"] if project["id"] == "flt" else CONFIG["project_sample_size"]
        if len(rows) > limit:
            mandatory = [row for row in rows if row["headline"]]
            candidates = [row for row in rows if not row["headline"]]
            rng.shuffle(candidates)
            rows = mandatory + candidates[:limit]
            rows.sort(key=lambda row: (row["file_path"], row["line"]))
        project_summary[project["id"]] = {
            "commit": commit,
            "lean_files": total_files,
            "lean_files_scanned": len(files),
            "declarations_extracted": extracted,
            "declarations_retained": len(rows),
            "headlines_retained": sum(row["headline"] for row in rows),
            "truncated_retained": sum(row["truncated"] for row in rows),
        }
        all_rows.extend(rows)
        print(f"{project['id']}: retained {len(rows):,}/{extracted:,}", flush=True)

    for index, row in enumerate(all_rows):
        row["i"] = index

    inputs = EXPERIMENT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "i", "project", "cohort", "repository", "commit", "file_path", "line",
        "kind", "name", "headline", "statement_sha256", "proof_sha256",
        "statement_characters", "proof_characters", "source_characters", "truncated",
    ]
    with (inputs / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps({key: row[key] for key in manifest_fields}, ensure_ascii=False) + "\n")

    total_characters = 0
    view_characters: dict[str, int] = {}
    for view, template in CONFIG["views"].items():
        count = 0
        with (inputs / f"{view}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                rendered = template.format(statement=row["statement"], proof=row["proof"])
                count += len(rendered)
                stream.write(json.dumps({"i": row["i"], "text": rendered}, ensure_ascii=False) + "\n")
        view_characters[view] = count
        total_characters += count

    embedding = CONFIG["embedding"]
    conservative_tokens = total_characters / embedding["cost_guard_characters_per_token"]
    projected_cost = conservative_tokens / 1_000_000 * embedding["estimated_price_usd_per_million_text_tokens"]
    preparation = {
        "experiment_id": CONFIG["experiment_id"],
        "declarations": len(all_rows),
        "projects": project_summary,
        "cohorts": dict(Counter(row["cohort"] for row in all_rows)),
        "view_characters": view_characters,
        "total_characters": total_characters,
        "conservative_token_estimate": conservative_tokens,
        "projected_cost_usd": projected_cost,
        "hard_cost_cap_usd": embedding["hard_cost_cap_usd"],
        "within_cap": projected_cost <= embedding["hard_cost_cap_usd"],
    }
    write_json(EXPERIMENT / "artifacts/preparation.json", preparation)
    print(json.dumps(preparation, indent=2))
    if not preparation["within_cap"]:
        raise SystemExit("Projected embedding cost exceeds the hard cap; AWS was not called.")


if __name__ == "__main__":
    main()
