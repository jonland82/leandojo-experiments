"""Print a compact summary of the generated topic experiment."""

import json


with open("out/stats.json", encoding="utf-8") as f:
    stats = json.load(f)

print("proofs", stats["n_theorems"], stats["source_counts"])
print("proof lengths", stats["proof_len"])
print(
    "layout PCA-3 explained variance",
    [round(v, 4) for v in stats["layout"]["pca3_evr"]],
)
for view_name in ("style", "domain"):
    view = stats[view_name]
    print(
        f"\n{view_name}: shape={view['matrix_shape']} topics={view['selected_topics']} "
        f"entropy={view['mean_mixture_entropy']:.3f} module_AMI={view['module_ami']:.3f}"
    )
    if "n_with_signal" in view:
        print("proofs with explicit premise signal", view["n_with_signal"])
    for topic in view["topics"]:
        print(
            f"  T{topic['id']:02d} dominant={topic['size']:4d} "
            f"mean_steps={topic['mean_len']:4.1f}  {topic['label']}"
        )
