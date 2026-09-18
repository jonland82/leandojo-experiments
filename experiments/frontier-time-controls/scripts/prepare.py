"""Prepare exact-snapshot Mathlib statement controls with a cost guard."""

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
CORPUS = ROOT / "data/leandojo_benchmark_4/leandojo_benchmark_4/corpus.jsonl"
REPO = ROOT / CONFIG["mathlib_repository"]

TARGET_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
    r"(?P<kind>theorem|lemma)\s+(?P<name>[^\s(:{\[]+)"
)
BOUNDARY_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|axiom|opaque|example)\b"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def old_statement_hashes() -> set[str]:
    hashes: set[str] = set()
    with CORPUS.open(encoding="utf-8") as stream:
        for line in stream:
            for premise in json.loads(line)["premises"]:
                code = premise.get("code", "").strip()
                if code:
                    hashes.add(digest(code))
    return hashes


def split_statement(block: str, search_start: int) -> str | None:
    """Split at the proof marker after the matched declaration header.

    Attributes may themselves contain ``:=`` (for example ``to_additive``
    options), so searching from the beginning silently truncates declarations.
    """
    marker = block.find(":=", search_start)
    if marker < 0:
        match = re.search(r"(?m)^\s*by\s*$", block[search_start:])
        if not match:
            return None
        marker = search_start + match.start()
    statement = block[:marker].strip()
    return statement or None


def extract(path: Path, old_hashes: set[str], revision: str) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    boundaries = list(BOUNDARY_RE.finditer(text))
    starts = [match.start() for match in boundaries]
    rows = []
    for match in TARGET_RE.finditer(text):
        position = bisect_right(starts, match.start())
        end = starts[position] if position < len(starts) else len(text)
        block = text[match.start():end]
        search_start = match.end() - match.start()
        statement = split_statement(block, search_start)
        if statement is None:
            continue
        statement_hash = digest(statement)
        rows.append({
            "revision": revision,
            "file_path": path.relative_to(REPO).as_posix(),
            "line": text.count("\n", 0, match.start()) + 1,
            "kind": match.group("kind"),
            "name": match.group("name").strip("`"),
            "statement": statement,
            "statement_sha256": statement_hash,
            "statement_characters": len(statement),
            "verbatim_in_2024_corpus": statement_hash in old_hashes,
        })
    return rows


def main() -> None:
    if not (REPO / ".git").exists():
        raise FileNotFoundError(
            f"missing {REPO}; clone https://github.com/leanprover-community/mathlib4.git there"
        )
    old_hashes = old_statement_hashes()
    revisions = sorted(set(CONFIG["projects"].values()))
    rng = random.Random(CONFIG["sample_seed"])
    retained = []
    summaries = {}
    for revision in revisions:
        git("checkout", "--detach", "--force", revision)
        date = git("show", "-s", "--format=%cI", revision)
        files = sorted((REPO / "Mathlib").rglob("*.lean"))
        rows = []
        for number, path in enumerate(files, 1):
            rows.extend(extract(path, old_hashes, revision))
            if number % 1000 == 0:
                print(f"{revision[:8]}: scanned {number:,}/{len(files):,} files", flush=True)
        novel = [row for row in rows if not row["verbatim_in_2024_corpus"]]
        rng.shuffle(novel)
        sample = novel[: CONFIG["control_sample_per_snapshot"]]
        sample.sort(key=lambda row: (row["file_path"], row["line"]))
        retained.extend(sample)
        summaries[revision] = {
            "commit_date": date,
            "lean_files": len(files),
            "declarations_extracted": len(rows),
            "verbatim_in_2024_corpus": sum(row["verbatim_in_2024_corpus"] for row in rows),
            "eligible_changed_or_new": len(novel),
            "retained": len(sample),
        }
        print(f"{revision[:8]}: retained {len(sample):,}/{len(novel):,} changed-or-new", flush=True)

    for index, row in enumerate(retained):
        row["i"] = index
    inputs = EXPERIMENT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "i", "revision", "file_path", "line", "kind", "name",
        "statement_sha256", "statement_characters", "verbatim_in_2024_corpus",
    ]
    total_characters = 0
    with (inputs / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest, \
            (inputs / "statement.jsonl").open("w", encoding="utf-8", newline="\n") as statements:
        for row in retained:
            manifest.write(json.dumps({key: row[key] for key in manifest_fields}, ensure_ascii=False) + "\n")
            rendered = CONFIG["view_template"].format(statement=row["statement"])
            total_characters += len(rendered)
            statements.write(json.dumps({"i": row["i"], "text": rendered}, ensure_ascii=False) + "\n")

    embedding = CONFIG["embedding"]
    tokens = total_characters / embedding["cost_guard_characters_per_token"]
    projected = tokens / 1_000_000 * embedding["estimated_price_usd_per_million_text_tokens"]
    preparation = {
        "experiment_id": CONFIG["experiment_id"],
        "reference_commit": CONFIG["reference_commit"],
        "reference_date": CONFIG["reference_date"],
        "old_statement_hashes": len(old_hashes),
        "declarations": len(retained),
        "snapshots": summaries,
        "project_snapshot_counts": dict(Counter(CONFIG["projects"].values())),
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
