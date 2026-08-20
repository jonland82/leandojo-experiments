"""Render the compact HTML report from frozen analysis artifacts."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def exemplar_rows(rows: list[dict], value_key: str) -> str:
    selected = rows[:2] + rows[-2:]
    labels = ["low", "low", "high", "high"]
    return "\n".join(
        "<tr>"
        f"<td><span class='tag {label}'>{label}</span></td>"
        f"<td><code>{escape(row['full_name'])}</code><small>{escape(row['file_path'])}</small></td>"
        f"<td>{fmt(row[value_key], 3)}</td>"
        "</tr>"
        for label, row in zip(labels, selected)
    )


def main() -> None:
    analysis = json.loads((EXPERIMENT / "artifacts/analysis.json").read_text(encoding="utf-8"))
    template = (EXPERIMENT / "scripts/report_template.html").read_text(encoding="utf-8")
    dependency = analysis["dependency_graph"]
    semantic = analysis["semantic"]
    complexity = analysis["complexity"]
    correlations = analysis["correlations"]["spearman"]
    sensitivity = analysis["sensitivity"]

    replacements = {
        "N_TARGETS": f"{analysis['n_targets']:,}",
        "N_DECLARATIONS": f"{dependency['exported_declarations']:,}",
        "N_EDGES": f"{dependency['edges']:,}",
        "DEP_MEDIAN": fmt(dependency["dependency_mass_log"]["median"]),
        "DIRECT_MEDIAN": fmt(dependency["direct_dependency_count"]["median"], 0),
        "DEPTH_MEDIAN": fmt(dependency["dependency_depth"]["median"], 0),
        "STATEMENT_THRESHOLD": fmt(semantic["statement"]["threshold"], 3),
        "PROOF_THRESHOLD": fmt(semantic["proof"]["threshold"], 3),
        "STATEMENT_NEFF": fmt(semantic["statement"]["effective_neighbors"]["median"], 1),
        "PROOF_NEFF": fmt(semantic["proof"]["effective_neighbors"]["median"], 1),
        "CROSS_VIEW_RHO": fmt(semantic["statement_proof_effective_neighbor_spearman"], 3),
        "COMPLEXITY_P95": fmt(complexity["score"]["p95"]),
        "DEP_COMPLEXITY_RHO": fmt(correlations[0][3], 3),
        "PROOF_COMPLEXITY_RHO": fmt(correlations[2][3], 3),
        "ALPHA_MIN_RHO": fmt(min(row["rho"] for row in sensitivity["dependency_alpha"]), 3),
        "THRESHOLD_MIN_RHO": fmt(
            min(
                min(row["statement_rho"], row["proof_rho"])
                for row in sensitivity["semantic_threshold"]
            ), 3,
        ),
        "COMPLEXITY_MIN_RHO": fmt(
            min(row["rho"] for row in sensitivity["complexity_leave_one_out"]), 3
        ),
        "DEPENDENCY_EXEMPLARS": exemplar_rows(
            analysis["exemplars"]["dependency_mass_log"], "dependency_mass_log"
        ),
        "SEMANTIC_EXEMPLARS": exemplar_rows(
            analysis["exemplars"]["statement_effective"], "statement_effective_neighbors"
        ),
        "COMPLEXITY_EXEMPLARS": exemplar_rows(
            analysis["exemplars"]["complexity"], "structural_complexity"
        ),
    }
    for key, value in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    output = EXPERIMENT / "index.html"
    output.write_text(template, encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
