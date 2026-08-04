"""Compute projection-free alignment statistics for one experiment artifact."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


def analyze(payload, n_permutations=20_000, seed=0):
    points = payload["points"]
    style = np.asarray([point["c"] for point in points], dtype=int)
    domain = np.asarray([point["domain_c"] for point in points], dtype=int)
    paired = (style >= 0) & (domain >= 0)
    style_paired = style[paired]
    domain_paired = domain[paired]
    n_style = int(payload["views"]["style"]["k"])
    n_domain = int(payload["views"]["domain"]["k"])

    table = np.bincount(
        domain_paired * n_style + style_paired,
        minlength=n_domain * n_style,
    ).reshape(n_domain, n_style)
    expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / table.sum()
    valid = expected > 0
    chi_squared = float(np.sum(((table - expected) ** 2 / expected)[valid]))
    cramers_v = float(
        np.sqrt(
            chi_squared
            / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))
        )
    )
    row_share = table / table.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = np.bincount(
            domain_paired * n_style + rng.permutation(style_paired),
            minlength=n_domain * n_style,
        ).reshape(n_domain, n_style)
        permuted_chi_squared = float(
            np.sum(((permuted - expected) ** 2 / expected)[valid])
        )
        exceedances += permuted_chi_squared >= chi_squared

    return {
        "n_proofs": len(points),
        "n_paired": int(paired.sum()),
        "style_topics": n_style,
        "domain_topics": n_domain,
        "style_unassigned": int((style < 0).sum()),
        "domain_unassigned": int((domain < 0).sum()),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(style_paired, domain_paired)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(style_paired, domain_paired)
        ),
        "pearson_chi_squared": chi_squared,
        "cramers_v": cramers_v,
        "within_domain_max_style_share": {
            "minimum": float(row_share.max(axis=1).min()),
            "median": float(np.median(row_share.max(axis=1))),
            "maximum": float(row_share.max(axis=1).max()),
        },
        "permutation_test": {
            "seed": seed,
            "permutations": n_permutations,
            "exceedances": int(exceedances),
            "corrected_monte_carlo_p": float(
                (exceedances + 1) / (n_permutations + 1)
            ),
        },
        "contingency_table_domain_by_style": table.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs_json", type=Path)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.proofs_json.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    result = analyze(payload, args.permutations, args.seed)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
