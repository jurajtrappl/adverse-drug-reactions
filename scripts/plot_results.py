"""Headline chart for the README: scaffold-split ROC-AUC of selected models -> results/leaderboard.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# (model id, label, input) in display order, bottom to top
SHOWN = [
    ("prior", "Coin flip (predict ADR frequency)", "none"),
    ("size+logreg", "Molecule size only", "structure"),
    ("chemprop", "Chemprop graph neural net", "structure"),
    ("knn5 tanimoto", "5 most similar drugs", "structure"),
    ("morgan+desc+mlp", "Multi-task MLP", "structure"),
    ("mol2vec+rf", "Pretrained Mol2vec + forest", "structure"),
    ("morgan+desc+rf", "Fingerprint + descriptors, forest", "structure"),
    ("ind+rf", "Indications only", "pharmacology"),
    ("atc+ind+rf", "ATC class + indications", "pharmacology"),
    ("structure rf + pharma rf", "Structure forest + pharmacology forest", "both"),
]
COLORS = {"none": "#8a8983", "structure": "#2a78d6", "pharmacology": "#eb6834", "both": "#1baf7a"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3dd"


def main():
    runs = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("phase*_runs.csv"))])
    auc = runs[runs.split == "scaffold"].groupby("model")["roc_auc"].agg(["mean"])

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (model, label, kind) in enumerate(SHOWN):
        m = auc.loc[model, "mean"]
        ax.plot([0.5, m], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(m, y, s=70, color=COLORS[kind], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m + 0.006, y, f"{m:.3f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(range(len(SHOWN)), [lbl for _, lbl, _ in SHOWN], fontsize=9, color=INK)
    ax.set_xlim(0.49, 0.76)
    ax.set_xlabel("Mean ROC-AUC over 27 ADR categories (scaffold split, 3 seeds; 0.5 = coin flip)",
                  fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], markersize=8, label=lbl)
               for k, lbl in [("structure", "Structure only"), ("pharmacology", "Pharmacology only"),
                              ("both", "Both")]]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("What predicts a drug's side effects?", loc="left", fontsize=12, color=INK,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    RES.mkdir(exist_ok=True)
    fig.savefig(RES / "leaderboard.png", facecolor=SURFACE)
    print("wrote", RES / "leaderboard.png")


if __name__ == "__main__":
    main()
