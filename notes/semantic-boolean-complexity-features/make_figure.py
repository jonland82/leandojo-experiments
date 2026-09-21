"""Build the compact importance figure used by the four-page note."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1] / "experiments" / "semantic-boolean-complexity-4bit" / "artifacts"
LANGUAGES = ("NAND", "NOR", "AND_OR_NOT")

individual = pd.read_csv(ARTIFACTS / "feature_permutation_importance.csv")
families = pd.read_csv(ARTIFACTS / "feature_family_permutation_importance.csv")
feature_ranking = pd.read_csv(ARTIFACTS / "feature_importance_ranking.csv").head(12).iloc[::-1]
family_ranking = pd.read_csv(ARTIFACTS / "feature_family_importance_ranking.csv").head(10).iloc[::-1]

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})
language_colors = {"NAND": "#40566A", "NOR": "#87939C", "AND_OR_NOT": "#25292C"}
offsets = {"NAND": -0.16, "NOR": 0.0, "AND_OR_NOT": 0.16}

feature_names = feature_ranking.feature.tolist()
feature_status = feature_ranking.set_index("feature").status
bar_colors = ["#B98249" if feature_status[name] == "new" else "#7A8C99"
              for name in feature_names]
axes[0].barh(np.arange(len(feature_names)), feature_ranking.importance_mean,
             color=bar_colors, alpha=0.55)
feature_values = individual.set_index(["feature", "language"])
for language in LANGUAGES:
    axes[0].scatter(
        [feature_values.loc[(name, language), "importance_mean"] for name in feature_names],
        np.arange(len(feature_names)) + offsets[language], color=language_colors[language],
        s=16, label=language.replace("_", "/"))
axes[0].set_yticks(
    np.arange(len(feature_names)),
    [f"{'NEW' if feature_status[name] == 'new' else 'USED'}  {name.replace('_', ' ')}"
     for name in feature_names], fontsize=7.5)
axes[0].set_title("Individual coordinates", loc="left", fontweight="bold")
axes[0].legend(frameon=False, loc="lower right", fontsize=7, handletextpad=0.3)

family_names = family_ranking.family.tolist()
axes[1].barh(np.arange(len(family_names)), family_ranking.importance_mean,
             color="#B98249", alpha=0.55)
family_values = families.set_index(["family", "language"])
for language in LANGUAGES:
    axes[1].scatter(
        [family_values.loc[(name, language), "importance_mean"] for name in family_names],
        np.arange(len(family_names)) + offsets[language], color=language_colors[language], s=16)
family_counts = family_ranking.set_index("family").features
axes[1].set_yticks(np.arange(len(family_names)),
                   [f"{name} ({family_counts[name]})" for name in family_names], fontsize=7.5)
axes[1].set_title("Joint feature families", loc="left", fontweight="bold")

for axis in axes:
    axis.axvline(0, color="#25292C", linewidth=0.7)
    axis.grid(axis="x", color="#D9DDDF", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel(r"held-out $R^2$ decrease after permutation", fontsize=8)
    axis.tick_params(axis="x", labelsize=7.5)

fig.tight_layout()
for extension in ("pdf", "png"):
    fig.savefig(HERE / f"importance_compact.{extension}", dpi=220, bbox_inches="tight")
plt.close(fig)
