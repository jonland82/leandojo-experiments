"""Exactly synthesize and count three-variable NAND formulas by semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
MASK = (1 << (1 << CONFIG["variables"])) - 1


def truth_table(predicate) -> int:
    output = 0
    for assignment in range(1 << CONFIG["variables"]):
        values = [(assignment >> index) & 1 for index in range(CONFIG["variables"])]
        output |= int(bool(predicate(*values))) << assignment
    return output


def nand(left: int, right: int) -> int:
    return (~(left & right)) & MASK


def and_cost(left: int, right: int) -> int:
    return 3 + 2 * left + 2 * right


def or_cost(left: int, right: int) -> int:
    return 3 + 2 * left + 2 * right


def combine_balanced(costs: list[int], operation) -> int:
    if len(costs) == 1:
        return costs[0]
    midpoint = len(costs) // 2
    return operation(combine_balanced(costs[:midpoint], operation),
                     combine_balanced(costs[midpoint:], operation))


def canonical_dnf_cost(table: int) -> int:
    satisfying = [assignment for assignment in range(1 << CONFIG["variables"])
                  if (table >> assignment) & 1]
    if not satisfying:
        return and_cost(0, 1)  # x0 AND (NOT x0)
    terms = []
    for assignment in satisfying:
        literal_costs = [0 if (assignment >> index) & 1 else 1
                         for index in range(CONFIG["variables"])]
        terms.append(combine_balanced(literal_costs, and_cost))
    return combine_balanced(terms, or_cost)


def canonical_truth_table_cost(table: int) -> int:
    """Smaller of canonical DNF(f) and NOT canonical DNF(NOT f)."""
    direct = canonical_dnf_cost(table)
    complement = canonical_dnf_cost((~table) & MASK)
    return min(direct, 1 + 2 * complement)


def main() -> None:
    variable_tables = [truth_table(lambda *values, i=index: values[i])
                       for index in range(CONFIG["variables"])]
    counts: list[dict[int, int]] = [{table: 1 for table in variable_tables}]
    witnesses: list[dict[int, str]] = [{table: f"x{index}"
                                       for index, table in enumerate(variable_tables)}]
    totals = [len(variable_tables)]
    minimum_size: dict[int, int] = {table: 0 for table in variable_tables}
    minimum_witness: dict[int, str] = dict(witnesses[0])
    for size in range(1, CONFIG["maximum_nand_gates"] + 1):
        size_counts: dict[int, int] = {}
        size_witnesses: dict[int, str] = {}
        for left_size in range(size):
            right_size = size - 1 - left_size
            for left, left_count in counts[left_size].items():
                for right, right_count in counts[right_size].items():
                    output = nand(left, right)
                    size_counts[output] = size_counts.get(output, 0) + left_count * right_count
                    if output not in size_witnesses:
                        size_witnesses[output] = (
                            f"NAND({witnesses[left_size][left]}, {witnesses[right_size][right]})")
        counts.append(size_counts)
        witnesses.append(size_witnesses)
        totals.append(sum(size_counts.values()))
        catalan = math.comb(2 * size, size) // (size + 1)
        expected_total = catalan * CONFIG["variables"] ** (size + 1)
        if totals[-1] != expected_total:
            raise RuntimeError(f"formula-count invariant failed at size {size}")
        for table, witness in size_witnesses.items():
            if table not in minimum_size:
                minimum_size[table] = size
                minimum_witness[table] = witness
        print(f"size {size}: {len(size_counts)} functions, {totals[-1]:,} formulas", flush=True)

    if len(minimum_size) != 256:
        missing = 256 - len(minimum_size)
        raise RuntimeError(f"maximum size failed to synthesize {missing} functions")
    targets = {
        "and3": truth_table(lambda x0, x1, x2: x0 and x1 and x2),
        "majority3": truth_table(lambda x0, x1, x2: x0 + x1 + x2 >= 2),
        "parity3": truth_table(lambda x0, x1, x2: (x0 ^ x1 ^ x2) == 1),
        "mux": truth_table(lambda x0, x1, x2: x1 if x0 else x2),
    }
    rows = []
    for table in range(256):
        size = minimum_size[table]
        correct_count = counts[size][table]
        probability = correct_count / totals[size]
        surprisal = -math.log2(probability)
        baseline = canonical_truth_table_cost(table)
        gain = baseline - size
        rows.append({"truth_table": table, "minimum_gates": size,
                     "minimum_formula_count": correct_count,
                     "all_formula_count_at_minimum": totals[size],
                     "minimum_probability": probability,
                     "boundary_surprisal_bits": surprisal,
                     "canonical_baseline_nand_gates": baseline,
                     "compression_gain": gain, "triad_score": gain * surprisal,
                     "witness": minimum_witness[table]})
    table = pd.DataFrame(rows)
    target_rows = []
    for name in CONFIG["targets"]:
        truth = targets[name]
        row = table.loc[table["truth_table"] == truth].iloc[0].to_dict()
        row["target"] = name
        target_rows.append(row)
    targets_table = pd.DataFrame(target_rows)
    size_surprisal = stats.spearmanr(table["minimum_gates"], table["boundary_surprisal_bits"])
    gain_surprisal = stats.spearmanr(table["compression_gain"], table["boundary_surprisal_bits"])
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifacts / "function_synthesis.csv", index=False)
    targets_table.to_csv(artifacts / "target_functions.csv", index=False)
    size_rows = [{"nand_gates": size, "semantic_functions": len(counts[size]),
                  "formula_count": totals[size]} for size in range(len(counts))]
    pd.DataFrame(size_rows).to_csv(artifacts / "size_counts.csv", index=False)
    result = {"experiment_id": CONFIG["experiment_id"], "functions_synthesized": len(minimum_size),
              "maximum_nand_gates": CONFIG["maximum_nand_gates"],
              "ordered_formula_count_invariant_verified": True,
              "largest_minimum_gate_count": int(table["minimum_gates"].max()),
              "minimum_gate_histogram": {str(int(key)): int(value) for key, value in
                                         table["minimum_gates"].value_counts().sort_index().items()},
              "spearman_minimum_size_vs_surprisal": {"rho": float(size_surprisal.statistic),
                                                       "pvalue": float(size_surprisal.pvalue)},
              "spearman_compression_gain_vs_surprisal": {"rho": float(gain_surprisal.statistic),
                                                           "pvalue": float(gain_surprisal.pvalue)},
              "targets": target_rows,
              "interpretation": "Exact synthesis verifies correctness and minimality, while semantic counts supply exact boundary rarity under the declared NAND-formula prior."}
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    histogram = table["minimum_gates"].value_counts().sort_index()
    axes[0].bar(histogram.index, histogram.values, color="#1d6996")
    axes[0].set_xlabel("exact minimum NAND gates")
    axes[0].set_ylabel("number of Boolean functions")
    axes[0].set_title("All 256 functions synthesized", loc="left")
    axes[1].scatter(table["minimum_gates"], table["boundary_surprisal_bits"],
                    s=15, alpha=0.5, color="#777777")
    for row in targets_table.itertuples():
        axes[1].scatter(row.minimum_gates, row.boundary_surprisal_bits, s=42, color="#d95f02")
        axes[1].annotate(row.target, (row.minimum_gates, row.boundary_surprisal_bits),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("exact minimum NAND gates")
    axes[1].set_ylabel("boundary surprisal (bits)")
    axes[1].set_title("Minimal semantics can also be rare", loc="left")
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "boolean_synthesis_creativity.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "boolean_synthesis_creativity.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = ["# Results: exact shortest Boolean-expression synthesis", "", "## Complete synthesis", "",
        f"All **256** three-variable Boolean functions are synthesized. The hardest functions require "
        f"**{int(table.minimum_gates.max())} NAND gates** in the ordered formula language. Correctness is "
        "verified by exact truth tables and minimality by exhaustive semantic dynamic programming.", "",
        "## Selected targets", "", "| Target | Minimum gates | Minimal formulas | Boundary surprisal | Canonical gain |",
        "|---|---:|---:|---:|---:|"]
    for row in targets_table.itertuples():
        lines.append(f"| {row.target} | {int(row.minimum_gates)} | {int(row.minimum_formula_count):,} | "
                     f"{row.boundary_surprisal_bits:.2f} bits | {int(row.compression_gain)} |")
    lines.extend(["", "## Interpretation", "",
        f"Across all functions, minimum formula size and boundary surprisal have Spearman "
        f"$\\rho={size_surprisal.statistic:.3f}$. Compression gain relative to the fixed canonical "
        f"compiler and surprisal have $\\rho={gain_surprisal.statistic:.3f}$. A shortest witness for each "
        "selected target therefore has mechanically verified correctness, compression against an explicit "
        "baseline, and rarity under an explicit syntax prior. Rare but uncompressed formulas and compressed "
        "but common semantics remain counterexamples to treating either component alone as creativity.", "",
        "![Boolean synthesis](figures/boolean_synthesis_creativity.png)"])
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
