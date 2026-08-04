"""Emit Markdown tables used in FINDINGS.md from the historical artifact."""

import json


with open("out/stats.json", encoding="utf-8") as f:
    stats = json.load(f)

for view_name in ("style", "domain"):
    view = stats[view_name]
    print(f"### {view_name} diagnostics\n")
    print("| topics | relative reconstruction error | stability | stability SD |")
    print("|---:|---:|---:|---:|")
    for row in view["diagnostics"]:
        marker = " **selected**" if row["k"] == view["selected_topics"] else ""
        print(
            f"| {row['k']}{marker} | {row['relative_reconstruction_error']:.4f} | "
            f"{row['stability']:.3f} | {row['stability_sd']:.3f} |"
        )
    print()

    print(f"### {view_name} topics\n")
    print("| # | dominant proofs | mean steps | label | top terms |")
    print("|---:|---:|---:|---|---|")
    for topic in view["topics"]:
        terms = ", ".join(f"`{term}`" for term in topic["top_terms"][:5])
        print(
            f"| {topic['id']} | {topic['size']} | {topic['mean_len']:.1f} | "
            f"{topic['label']} | {terms} |"
        )
    print()
