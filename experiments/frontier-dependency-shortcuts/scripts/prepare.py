"""Build conservative lexical dependency graphs for frontier projects."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "experiments/frontier-formalizations"
SOURCES = ROOT / "data/frontier_sources"
CONFIG = json.loads((FRONTIER / "config.json").read_text(encoding="utf-8"))

DECL_RE = re.compile(
    r"(?m)^(?:\s*@\[[^\n]*?\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial|nonrec)\s+)*"
    r"(?P<kind>theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|axiom|opaque)"
    r"(?:\s+(?P<name>[^\s(:{\[]+))?"
)
TOKEN_RE = re.compile(r"(?<![\w'])([^\W\d][\w']*(?:\.[^\W\d][\w']*)*)(?![\w'])")
NAMESPACE_RE = re.compile(r"^\s*namespace(?:\s+([^\s]+))?\s*$")
SECTION_RE = re.compile(r"^\s*section(?:\s+([^\s]+))?\s*$")
END_RE = re.compile(r"^\s*end(?:\s+([^\s]+))?\s*$")


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def namespace_by_line(lines: list[str]) -> list[str]:
    namespaces: list[str] = []
    scopes: list[tuple[str, int]] = []
    result = [""] * (len(lines) + 1)
    for number, line in enumerate(lines, 1):
        result[number] = ".".join(namespaces)
        match = NAMESPACE_RE.match(line)
        if match:
            parts = (match.group(1) or "").split(".") if match.group(1) else []
            scopes.append(("namespace", len(parts)))
            namespaces.extend(parts)
            continue
        if SECTION_RE.match(line):
            scopes.append(("section", 0))
            continue
        if END_RE.match(line) and scopes:
            kind, count = scopes.pop()
            if kind == "namespace" and count:
                del namespaces[-count:]
    return result


def split_body(block: str, header_end: int) -> tuple[int, int]:
    marker = block.find(":=", header_end)
    if marker >= 0:
        return marker, marker + 2
    by = re.search(r"(?m)^\s*by\b", block[header_end:])
    if by:
        start = header_end + by.start()
        return start, header_end + by.end()
    where = re.search(r"(?m)^\s*where\b", block[header_end:])
    if where:
        start = header_end + where.start()
        return start, header_end + where.end()
    return len(block), len(block)


def scan_project(project: str, excluded: list[str], headlines: list[dict]) -> tuple[list[dict], dict]:
    repo = SOURCES / project
    nodes: list[dict] = []
    by_location = {(row["file_path"], row["line"]): row for row in headlines}
    files = [path for path in repo.rglob("*.lean") if not any(
        path.relative_to(repo).as_posix().startswith(prefix) for prefix in excluded
    )]
    for path in sorted(files):
        relative = path.relative_to(repo).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        newline_positions = [match.start() for match in re.finditer("\n", text)]
        namespaces = namespace_by_line(lines)
        matches = list(DECL_RE.finditer(text))
        for index, match in enumerate(matches):
            name = (match.group("name") or f"_anonymous_{index}").strip("`")
            line = bisect_right(newline_positions, match.start()) + 1
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start():end]
            statement_end, body_start = split_body(block, match.end() - match.start())
            namespace = namespaces[min(line, len(lines))]
            full_name = name if "." in name or not namespace else f"{namespace}.{name}"
            headline = by_location.get((relative, line))
            if headline is None:
                candidates = [row for row in headlines if row["file_path"] == relative
                              and row["name"].split(".")[-1] == name.split(".")[-1]
                              and abs(row["line"] - line) <= 5]
                headline = candidates[0] if len(candidates) == 1 else None
            nodes.append({
                "id": len(nodes), "project": project, "file_path": relative, "line": line,
                "file_lines": len(lines), "file_declarations": len(matches),
                "relative_position": line / max(1, len(lines)), "kind": match.group("kind"),
                "name": name, "full_name": full_name,
                "statement_characters": statement_end,
                "proof_characters": max(0, len(block) - body_start),
                "source": block, "headline": bool(headline),
                "headline_name": headline["name"] if headline else None,
            })

    full_lookup: dict[str, set[int]] = defaultdict(set)
    short_lookup: dict[str, set[int]] = defaultdict(set)
    file_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    for node in nodes:
        full_lookup[node["full_name"]].add(node["id"])
        full_lookup[node["name"]].add(node["id"])
        short = node["name"].split(".")[-1]
        short_lookup[short].add(node["id"])
        file_lookup[(node["file_path"], short)].append(node["id"])

    edge_count = 0
    resolved_tokens = 0
    for node in nodes:
        dependencies: set[int] = set()
        for token in set(TOKEN_RE.findall(node.pop("source"))):
            candidates = full_lookup.get(token, set())
            if len(candidates) == 1:
                target = next(iter(candidates))
            else:
                short = token.split(".")[-1]
                candidates = short_lookup.get(short, set())
                if len(candidates) == 1:
                    target = next(iter(candidates))
                else:
                    same_file = file_lookup.get((node["file_path"], short), [])
                    position = bisect_right(same_file, node["id"] - 1)
                    target = same_file[position - 1] if position else -1
            if target >= 0 and target != node["id"]:
                dependencies.add(target)
                resolved_tokens += 1
        node["dependencies"] = sorted(dependencies)
        edge_count += len(dependencies)
    return nodes, {"files": len(files), "nodes": len(nodes), "edges": edge_count,
                   "resolved_tokens": resolved_tokens,
                   "headlines": sum(node["headline"] for node in nodes)}


def main() -> None:
    manifest = jsonl(FRONTIER / "inputs/manifest.jsonl")
    headline_rows: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        if row.get("headline"):
            headline_rows[row["project"]].append(row)
    all_nodes = []
    summaries = {}
    for project_config in CONFIG["projects"]:
        project = project_config["id"]
        nodes, summary = scan_project(
            project, project_config.get("exclude_prefixes", []), headline_rows[project]
        )
        offset = len(all_nodes)
        for node in nodes:
            node["id"] += offset
            node["dependencies"] = [value + offset for value in node["dependencies"]]
        all_nodes.extend(nodes)
        summaries[project] = summary
        print(f"{project}: {summary['nodes']:,} declarations, {summary['edges']:,} edges, "
              f"{summary['headlines']} headlines", flush=True)
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "graph_nodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for node in all_nodes:
            stream.write(json.dumps(node, ensure_ascii=False) + "\n")
    result = {"experiment_id": "frontier-dependency-shortcuts", "projects": summaries,
              "total_nodes": len(all_nodes),
              "total_edges": sum(row["edges"] for row in summaries.values()),
              "headline_nodes": sum(row["headlines"] for row in summaries.values())}
    (artifacts / "preparation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
