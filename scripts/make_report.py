"""Turn results/phase1_*.csv into results/REPORT.md."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BEST = "morgan+desc+rf"

DESCRIPTIONS = {
    "prior": "Predict each ADR's training frequency (coin flip by construction)",
    "size+logreg": "Heavy-atom count + SMILES length, logistic regression",
    "knn5 tanimoto": "Copy labels of the 5 most similar training drugs (no training)",
    "morgan+logreg": "Morgan fingerprint, logistic regression",
    "morgan+rf": "Morgan fingerprint, random forest per ADR",
    "desc+rf": "RDKit descriptors, random forest per ADR",
    "morgan+desc+rf": "Morgan + descriptors, random forest per ADR",
    "morgan+desc+rf-multi": "Morgan + descriptors, one forest for all 27 ADRs",
    "morgan+desc+lgbm": "Morgan + descriptors, LightGBM per ADR",
}


def main():
    s = pd.read_csv(RES / "phase1_summary.csv")
    wide = s.pivot(index="model", columns="split", values=["roc_auc", "roc_auc_std", "pr_auc_lift"])
    order = [m for m in DESCRIPTIONS if m in wide.index]
    lines = ["# Phase 1 results", "",
             "Mean over 27 ADRs of the per-ADR ROC-AUC, averaged over 5 folds and 3 seeds.",
             "PR-AUC lift = PR-AUC minus the ADR's prevalence (0 = no better than guessing).",
             "Scaffold split = no Murcko scaffold is shared between train and test (the honest",
             "'new drug' setting). Regenerate with `python scripts/run_phase1.py && "
             "python scripts/make_report.py`.", "",
             "| Model | What it is | ROC-AUC scaffold | ROC-AUC random | PR-AUC lift scaffold |",
             "| --- | --- | --- | --- | --- |"]
    for m in order:
        r = wide.loc[m]
        name = f"**{m}**" if m == BEST else m
        lines.append(f"| {name} | {DESCRIPTIONS[m]} | {r[('roc_auc', 'scaffold')]:.3f} "
                     f"± {r[('roc_auc_std', 'scaffold')]:.3f} | {r[('roc_auc', 'random')]:.3f} | "
                     f"{r[('pr_auc_lift', 'scaffold')]:+.3f} |")

    a = pd.read_csv(RES / "phase1_per_adr.csv")
    a = (a[(a.model == BEST) & (a.split == "scaffold")]
         .groupby("adr").agg(roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
                             prevalence=("prevalence", "first"))
         .sort_values("roc_auc", ascending=False))
    lines += ["", f"## Per ADR: `{BEST}`, scaffold split", "",
              "| ADR | Positive drugs | ROC-AUC | PR-AUC |", "| --- | --- | --- | --- |"]
    for adr, r in a.iterrows():
        lines.append(f"| {adr} | {r.prevalence:.0%} | {r.roc_auc:.3f} | {r.pr_auc:.3f} |")
    (RES / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
