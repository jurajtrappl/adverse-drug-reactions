"""Turn results/phase*_*.csv into results/REPORT.md."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASELINE = "morgan+desc+rf"  # Phase 1 winner: every Phase 2 model is compared against it

DESCRIPTIONS = {
    # Phase 1
    "prior": "Predict each ADR's training frequency (coin flip by construction)",
    "size+logreg": "Heavy-atom count + SMILES length, logistic regression",
    "knn5 tanimoto": "Copy labels of the 5 most similar training drugs (no training)",
    "morgan+logreg": "Morgan fingerprint, logistic regression",
    "morgan+rf": "Morgan fingerprint, random forest per ADR",
    "desc+rf": "RDKit descriptors, random forest per ADR",
    "morgan+desc+rf": "Morgan + descriptors, random forest per ADR",
    "morgan+desc+rf-multi": "Morgan + descriptors, one forest for all 27 ADRs (no class weights)",
    "morgan+desc+lgbm": "Morgan + descriptors, LightGBM per ADR",
    # Phase 2
    "morgan+desc+mlp": "Morgan + descriptors, multi-task MLP (27 outputs)",
    "mol2vec+logreg": "Pretrained Mol2vec embedding, logistic regression",
    "mol2vec+rf": "Pretrained Mol2vec embedding, random forest per ADR",
    "mol2vec+mlp": "Pretrained Mol2vec embedding, multi-task MLP",
    "mlp + rf ensemble": "Average of the multi-task MLP and the random forest",
    "chemprop": "Chemprop D-MPNN graph network, multi-task",
    "chemprop+desc": "Chemprop D-MPNN + RDKit descriptors, multi-task",
    # Phase 3
    "count-only": "Predicted ADR count only (from structure), logistic regression per ADR",
    "count-stacked rf": "Morgan + descriptors + predicted ADR count, random forest",
    "atc+rf": "WHO ATC class (level 1+2, 45% of drugs matched), random forest",
    "ind+rf": "SIDER indications (what the drug treats), random forest",
    "atc+ind+rf": "ATC + indications, random forest",
    "morgan+desc+atc+ind+rf": "Structure + ATC + indications in one random forest",
    "structure rf + pharma rf": "Average of a structure forest and a pharmacology forest",
}

PHARMA_MODELS = {"atc+rf", "ind+rf", "atc+ind+rf", "morgan+desc+atc+ind+rf",
                 "structure rf + pharma rf"}


def uses(model):
    if model == "prior":
        return "—"
    if model in PHARMA_MODELS:
        return "structure + pharma" if "morgan" in model or "structure" in model else "pharma"
    return "structure"


def load(phase):
    runs = RES / f"{phase}_runs.csv"
    if not runs.exists():
        return None, None
    r = pd.read_csv(runs)
    r["phase"] = phase
    a = pd.read_csv(RES / f"{phase}_per_adr.csv")
    return r, a


def main():
    runs, adrs = zip(*[load(p) for p in ("phase1", "phase2", "phase3")])
    runs = pd.concat([r for r in runs if r is not None], ignore_index=True)
    adrs = pd.concat([a for a in adrs if a is not None], ignore_index=True)
    runs = runs.drop_duplicates(["split", "seed", "model"], keep="first")

    lines = ["# Results", "",
             "Mean over 27 ADRs of the per-ADR ROC-AUC, averaged over 5 folds and 3 seeds.",
             "PR-AUC lift = PR-AUC minus the ADR's prevalence (0 = no better than guessing).",
             "Scaffold split = no Murcko scaffold is shared between train and test (the honest",
             "'new drug' setting). All models use the same folds (`data/splits/`).",
             "Input 'pharma' = ATC class and/or SIDER indications: known only for marketed drugs,",
             "so those rows explain ADRs rather than predict them for a new molecule.", "",
             "Regenerate: `python scripts/run_benchmark.py --set phase1`, "
             "`--set phase2`, `--set phase3`, then `python scripts/make_report.py`.", ""]

    # ---- leaderboard on the scaffold split, both phases --------------------------------
    sc = runs[runs.split == "scaffold"]
    base = sc[sc.model == BASELINE].set_index("seed")["roc_auc"]
    sc = sc.assign(delta=sc.roc_auc - sc.seed.map(base))
    board = (sc.groupby(["model", "phase"])
             .agg(roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"), lift=("pr_auc_lift", "mean"),
                  d=("delta", "mean"), d_lo=("delta", "min"), d_hi=("delta", "max"))
             .reset_index().sort_values("roc", ascending=False))
    lines += ["## Scaffold-split leaderboard (all phases)", "",
              f"Δ vs RF = difference to `{BASELINE}` on the same seed, mean [min, max] over the 3 seeds.",
              "", "| Model | Phase | Input | What it is | ROC-AUC | Δ vs RF | PR-AUC lift |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in board.iterrows():
        name = f"**{r.model}**" if r.model == board.iloc[0].model else r.model
        delta = "—" if r.model == BASELINE else f"{r.d:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]"
        lines.append(f"| {name} | {r.phase[-1]} | {uses(r.model)} | {DESCRIPTIONS.get(r.model, '')} | "
                     f"{r.roc:.3f} ± {r.roc_sd:.3f} | {delta} | {r.lift:+.3f} |")

    # ---- Phase 1: scaffold vs random ----------------------------------------------------
    p1 = runs[runs.phase == "phase1"]
    wide = p1.groupby(["model", "split"])["roc_auc"].mean().unstack()
    lines += ["", "## Scaffold vs random split (Phase 1)", "",
              "The random split lets near-identical drugs sit on both sides, so it flatters every model.",
              "", "| Model | ROC-AUC scaffold | ROC-AUC random |", "| --- | --- | --- |"]
    for m in [m for m in DESCRIPTIONS if m in wide.index]:
        lines.append(f"| {m} | {wide.loc[m, 'scaffold']:.3f} | {wide.loc[m, 'random']:.3f} |")

    # ---- per ADR: best model vs the RF baseline ------------------------------------------
    best = board.iloc[0].model
    per = (adrs[adrs.split == "scaffold"].groupby(["model", "adr"])
           .agg(roc=("roc_auc", "mean"), pr=("pr_auc", "mean"), prev=("prevalence", "first")))
    cols = [BASELINE] if best == BASELINE else [best, BASELINE]
    t = pd.concat({m: per.loc[m] for m in cols}, axis=1).sort_values((cols[0], "roc"), ascending=False)
    head = " | ".join(f"ROC-AUC `{m}`" for m in cols)
    lines += ["", "## Per ADR, scaffold split", "",
              f"| ADR | Positive drugs | {head} | PR-AUC `{cols[0]}` |",
              "| --- | --- | " + " | ".join("---" for _ in cols) + " | --- |"]
    for adr_name, r in t.iterrows():
        rocs = " | ".join(f"{r[(m, 'roc')]:.3f}" for m in cols)
        lines.append(f"| {adr_name} | {r[(cols[0], 'prev')]:.0%} | {rocs} | {r[(cols[0], 'pr')]:.3f} |")

    (RES / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
