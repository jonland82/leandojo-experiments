"""Exactly count correct four-input sorting networks by comparator length."""

from __future__ import annotations

import itertools
import json
import math
import functools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NAVY = "#40566A"
SLATE = "#87939C"
CHARCOAL = "#25292C"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.edgecolor": CHARCOAL, "axes.labelcolor": CHARCOAL,
                     "text.color": CHARCOAL, "xtick.color": CHARCOAL,
                     "ytick.color": CHARCOAL, "savefig.facecolor": "white"})


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))


def apply_comparator(values: tuple[int, ...], comparator: tuple[int, int]) -> tuple[int, ...]:
    i, j = comparator
    if values[i] <= values[j]:
        return values
    output = list(values)
    output[i], output[j] = output[j], output[i]
    return tuple(output)


def main() -> None:
    n = CONFIG["n"]
    permutations = list(itertools.permutations(range(n)))
    permutation_index = {value: index for index, value in enumerate(permutations)}
    comparators = list(itertools.combinations(range(n), 2))
    transitions = [[permutation_index[apply_comparator(value, comparator)]
                    for value in permutations] for comparator in comparators]
    initial = tuple(range(len(permutations)))
    sorted_index = permutation_index[tuple(range(n))]
    correct_state = tuple([sorted_index] * len(permutations))
    counts: dict[tuple[int, ...], int] = {initial: 1}
    representatives: dict[tuple[int, ...], tuple[int, ...]] = {initial: ()}
    rows = []
    first_witness: tuple[int, ...] | None = None
    minimum_length: int | None = None
    minimum_count: int | None = None
    for length in range(CONFIG["maximum_length"] + 1):
        correct = counts.get(correct_state, 0)
        total = len(comparators) ** length
        probability = correct / total
        surprisal = -math.log2(probability) if probability else math.inf
        if correct and minimum_length is None:
            minimum_length = length
            minimum_count = correct
            first_witness = representatives[correct_state]
        padded_lower_bound = 0
        if minimum_length is not None and length >= minimum_length:
            padded_lower_bound = minimum_count * len(comparators) ** (length - minimum_length)
        compression_gain = max(0, len(CONFIG["baseline_comparators"]) - length)
        rows.append({"length": length, "total_sequences": total,
                     "semantic_states": len(counts), "correct_sequences": correct,
                     "correct_probability": probability, "correct_surprisal_bits": surprisal,
                     "padded_optimum_lower_bound": padded_lower_bound,
                     "compression_gain": compression_gain,
                     "triad_score_for_correct_sequence": compression_gain * surprisal if correct else 0.0})
        if length == CONFIG["maximum_length"]:
            break
        new_counts: dict[tuple[int, ...], int] = defaultdict(int)
        new_representatives: dict[tuple[int, ...], tuple[int, ...]] = {}
        for state, multiplicity in counts.items():
            prefix = representatives[state]
            for comparator_index, transition in enumerate(transitions):
                target = tuple(transition[value] for value in state)
                new_counts[target] += multiplicity
                new_representatives.setdefault(target, prefix + (comparator_index,))
        counts = dict(new_counts)
        representatives = new_representatives
        print(f"length {length + 1}: {len(counts):,} semantic states", flush=True)

    if first_witness is None or minimum_length is None:
        raise RuntimeError("no sorting network found")
    witness = [comparators[index] for index in first_witness]
    baseline = [tuple(pair) for pair in CONFIG["baseline_comparators"]]
    for sequence, label in ((witness, "minimum witness"), (baseline, "baseline")):
        if not all(
            (lambda output: output == tuple(range(n)))(
                functools.reduce(apply_comparator, sequence, permutation)
            ) for permutation in permutations
        ):
            raise RuntimeError(f"{label} failed exhaustive verification")
    brute_minimum_count = 0
    for indices in itertools.product(range(len(comparators)), repeat=minimum_length):
        sequence = [comparators[index] for index in indices]
        if all(functools.reduce(apply_comparator, sequence, permutation)
               == tuple(range(n)) for permutation in permutations):
            brute_minimum_count += 1
    if brute_minimum_count != minimum_count:
        raise RuntimeError("dynamic and brute-force minimum counts disagree")

    table = pd.DataFrame(rows)
    artifacts = EXPERIMENT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifacts / "length_counts.csv", index=False)
    finite = table[table["correct_sequences"] > 0]
    result = {"experiment_id": CONFIG["experiment_id"], "inputs_verified": len(permutations),
              "comparators": [list(pair) for pair in comparators],
              "minimum_length": minimum_length, "minimum_correct_sequences": minimum_count,
              "minimum_correct_probability": float(finite.iloc[0]["correct_probability"]),
              "minimum_correct_surprisal_bits": float(finite.iloc[0]["correct_surprisal_bits"]),
              "minimum_witness": [list(pair) for pair in witness],
              "brute_force_minimum_count_verified": True,
              "baseline_length": len(baseline), "baseline_verified": True,
              "maximum_length": CONFIG["maximum_length"],
              "interpretation": "The conjunction of correctness and one-comparison compression occurs only at the exact five-comparator boundary; appending arbitrary comparators preserves correctness but destroys compression."}
    (artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].plot(finite["length"], finite["correct_probability"], marker="o", color=NAVY)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("comparator sequence length")
    axes[0].set_ylabel("exact probability of sorting all inputs")
    axes[0].set_title("Correctness is rare at the boundary", loc="left")
    axes[0].axvline(minimum_length, color=CHARCOAL, linewidth=0.8, linestyle="--")
    axes[1].plot(finite["length"], finite["correct_sequences"], marker="o",
                 label="all correct sequences", color=NAVY)
    axes[1].plot(finite["length"], finite["padded_optimum_lower_bound"], marker="s",
                 label="padded optimal prefixes", color=SLATE)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("comparator sequence length")
    axes[1].set_ylabel("exact count")
    axes[1].set_title("Correct but redundant forms proliferate", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figures = EXPERIMENT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "sorting_creativity.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / "sorting_creativity.pdf", bbox_inches="tight")
    plt.close(fig)

    first = finite.iloc[0]
    lines = ["# Results: exact four-input sorting-network creativity test", "", "## Boundary result", "",
        f"All **{len(permutations)}** input permutations were verified. The minimum correct network has "
        f"**{minimum_length} comparators**; exactly **{minimum_count:,}** of the "
        f"**{int(first.total_sequences):,}** length-{minimum_length} sequences are correct. Thus",
        "", f"\\[P(\\text{{correct}}\\mid L={minimum_length})={first.correct_probability:.8f},"
        f"\\qquad -\\log_2 P={first.correct_surprisal_bits:.3f}\\text{{ bits}}.\\]", "",
        f"A lexicographically first minimum witness is `{witness}`. The fixed insertion-network baseline "
        f"uses {len(baseline)} comparisons, so the witness has verified structure preservation, a one-comparison "
        "compression gain, and exact boundary surprisal.", "", "## Interpretation", "",
        "Appending any comparator to an already sorted output preserves correctness. The exact counts therefore "
        "separate genuinely boundary-compressed networks from large families of correct but redundant programs. "
        "This verifies the triad relative to the declared network language and uniform conditional prior; it does "
        "not establish a representation-independent notion of creativity.", "",
        "![Sorting-network counts](figures/sorting_creativity.png)"]
    (EXPERIMENT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
