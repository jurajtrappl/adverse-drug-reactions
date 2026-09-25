"""Phase 4 benchmark: specific side effects (MedDRA preferred terms) instead of 27 organ classes.

Same drugs, same scaffold folds (restricted to the 1,422 drugs with a STITCH id), same
features. One multi-output random forest per feature set keeps 579 labels affordable.

Usage: python scripts/run_fine.py            -> results/phase4_*.csv and results/FINE_LABELS.md
       python scripts/run_fine.py --report  -> rebuild FINE_LABELS.md from the saved CSVs
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from adr import data, fine_labels, pharma, splits  # noqa: E402
from adr.evaluate import cross_validate  # noqa: E402
from adr.models import N_STRUCTURE, BlockAverage, MultiOutputRF, Prior, TanimotoKNN, logreg  # noqa: E402
from run_benchmark import featurize  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "results"
SEEDS = (0, 1, 2)


class Cols:
    """Wrap a model so it only sees a column slice of X."""

    def __init__(self, sl, model):
        self.sl, self.model = sl, model

    def fit(self, X, Y):
        self.model.fit(X[:, self.sl], Y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, self.sl])


def doc_flags(df):
    ind, _ = pharma.indications(df)
    atc, _ = pharma.atc(df)
    return np.column_stack([ind[:, -1], atc[:, -1], ind[:, :-1].sum(1)])


def experiments(n_struct, n_all):
    S, P = slice(0, n_struct), slice(n_struct, n_all)
    return {
        "prior": ("all", lambda s: Prior()),
        "documentation flags + logreg": ("flags", lambda s: logreg(seed=s)),
        "knn5 tanimoto": ("morgan", lambda s: TanimotoKNN(5)),
        "structure rf": ("all", lambda s: Cols(S, MultiOutputRF(seed=s))),
        "pharma rf": ("all", lambda s: Cols(P, MultiOutputRF(seed=s))),
        "structure rf + pharma rf": ("all", lambda s: BlockAverage(
            n_struct, lambda: MultiOutputRF(seed=s), lambda: MultiOutputRF(seed=s))),
    }


def evaluate(make, X, Y, folds):
    """Per-term ROC-AUC and PR-AUC, averaged over folds (same protocol as Phases 1-3)."""
    roc, pr, _ = cross_validate(make, X, Y, folds)
    return np.nanmean(roc, 0), np.nanmean(pr, 0)


def pharma_without_label_terms(df, terms):
    """ATC + indications, minus indication terms that are also side-effect labels.

    SIDER text mining lists 15% of a drug's indication terms among its side effects too
    (e.g. an antihypertensive with "Hypertension" in both). An indication column with the
    same MedDRA term as a label would partly hand the label to the model, so it is dropped.
    """
    ind, names = pharma.indications(df)
    atc, _ = pharma.atc(df)
    labels = set(terms)
    cols = [j for j, n in enumerate(names) if n.removeprefix("ind:") not in labels]
    return np.hstack([atc, ind[:, cols]]), len(names) - len(cols)


def main():
    df, _ = data.load()
    keep, Y, terms = fine_labels.load(df)
    pharma_x, n_dropped = pharma_without_label_terms(df, terms)
    print(f"dropped {n_dropped} indication columns that share a term with a label", flush=True)
    structure = featurize("morgan+desc", df)
    assert structure.shape[1] == N_STRUCTURE
    feats = {"all": np.hstack([structure, pharma_x])[keep],
             "morgan": featurize("morgan", df)[keep],
             "flags": doc_flags(df)[keep]}
    exps = experiments(N_STRUCTURE, feats["all"].shape[1])
    prev = Y.mean(0)
    print(f"{len(Y)} drugs, {len(terms)} terms (listed for >= {fine_labels.MIN_DRUGS} drugs)", flush=True)

    rows, per_term = [], []
    for seed in SEEDS:
        folds = fine_labels.restrict_folds(
            splits.load_folds(ROOT / "data" / "splits" / f"scaffold_seed{seed}.csv"), keep)
        for name, (fname, factory) in exps.items():
            t0 = time.time()
            roc, pr = evaluate(lambda: factory(seed), feats[fname], Y, folds)
            rows.append(dict(seed=seed, model=name, roc_auc=roc.mean(), pr_auc_lift=(pr - prev).mean(),
                             seconds=round(time.time() - t0, 1)))
            per_term += [dict(seed=seed, model=name, term=t, prevalence=p, roc_auc=r, pr_auc=q)
                         for t, p, r, q in zip(terms, prev, roc, pr)]
            print(f"seed={seed} {name:28s} ROC-AUC={roc.mean():.3f} ({rows[-1]['seconds']}s)", flush=True)
    runs, pt = pd.DataFrame(rows), pd.DataFrame(per_term)
    RES.mkdir(exist_ok=True)
    runs.to_csv(RES / "phase4_runs.csv", index=False)
    pt.to_csv(RES / "phase4_per_term.csv", index=False)
    write_report(runs, pt, Y, terms, df)


def write_report(runs, pt, Y, terms, df):
    s = runs.groupby("model", sort=False).agg(roc=("roc_auc", "mean"), sd=("roc_auc", "std"),
                                              lift=("pr_auc_lift", "mean"))
    t = pt.groupby(["model", "term"]).agg(roc=("roc_auc", "mean"), prev=("prevalence", "first"))
    wide = t["roc"].unstack(0)
    prev = t.loc["prior", "prev"]
    ev = PCA().fit(Y).explained_variance_ratio_
    pc1 = PCA(1).fit_transform(Y)[:, 0]
    r_count = abs(np.corrcoef(pc1, Y.sum(1))[0, 1])

    out = ["# Phase 4: specific side effects", "",
           f"{len(Y)} drugs, {len(terms)} MedDRA preferred terms listed for at least "
           f"{Y.sum(0).min()} drugs (prevalence {prev.min():.1%}-{prev.max():.0%}). Scaffold split, "
           "5 folds, 3 seeds; ROC-AUC per term and fold, averaged. Indication features that share "
           "a MedDRA term with any label are removed (see `pharma_without_label_terms`).", "",
           "## Overall", "",
           "| Model | Mean ROC-AUC | PR-AUC lift |", "| --- | --- | --- |"]
    out += [f"| {m} | {r.roc:.3f} ± {r.sd:.3f} | {r.lift:+.3f} |" for m, r in s.iterrows()]

    bins = pd.cut(prev, [0, 0.1, 0.25, 0.5, 1.0], labels=["<10%", "10-25%", "25-50%", ">50%"])
    cols = ["structure rf", "pharma rf", "structure rf + pharma rf"]
    out += ["", "## By how common the side effect is", "",
            "| Share of drugs listing the term | Terms | " + " | ".join(cols) + " |",
            "| --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    for b in bins.cat.categories:
        idx = bins.index[bins == b]
        out.append(f"| {b} | {len(idx)} | " + " | ".join(f"{wide.loc[idx, c].mean():.3f}" for c in cols) + " |")

    out += ["", "## Is it still one factor?", "",
            f"PC1 of the {len(terms)}-term matrix explains {ev[0]:.0%} of the variance "
            f"(27 organ classes: 33%) and correlates with the number of listed terms at "
            f"|r| = {r_count:.2f}. The 'long package insert' factor is still there.", ""]

    d = wide.assign(gap=wide["structure rf"] - wide["pharma rf"], prev=prev)
    best_struct = d.sort_values("structure rf", ascending=False).head(15)
    out += ["## Where structure is strongest", "",
            "Terms with the highest structure-only ROC-AUC. Most look like class effects of drug "
            "families a fingerprint recognises (antipsychotics, anticholinergics, antibiotics, "
            "steroid hormones).", "",
            "| Term | Drugs listing it | Structure | Pharmacology | Structure - pharma |",
            "| --- | --- | --- | --- | --- |"]
    out += [f"| {i} | {r.prev:.0%} | {r['structure rf']:.3f} | {r['pharma rf']:.3f} | {r.gap:+.3f} |"
            for i, r in best_struct.iterrows()]
    n_better = int((d.gap > 0).sum())
    out += ["", f"Structure beats pharmacology on {n_better} of {len(d)} terms. The 10 largest "
            "structure advantages:", "",
            "| Term | Drugs listing it | Structure | Pharmacology |", "| --- | --- | --- | --- |"]
    out += [f"| {i} | {r.prev:.0%} | {r['structure rf']:.3f} | {r['pharma rf']:.3f} |"
            for i, r in d.sort_values("gap", ascending=False).head(10).iterrows()]
    out += ["", "## Where pharmacology dominates", "",
            "| Term | Drugs listing it | Structure | Pharmacology |", "| --- | --- | --- | --- |"]
    out += [f"| {i} | {r.prev:.0%} | {r['structure rf']:.3f} | {r['pharma rf']:.3f} |"
            for i, r in d.sort_values("gap").head(10).iterrows()]
    (RES / "FINE_LABELS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def report_only():
    df, _ = data.load()
    keep, Y, terms = fine_labels.load(df)
    write_report(pd.read_csv(RES / "phase4_runs.csv"), pd.read_csv(RES / "phase4_per_term.csv"),
                 Y, terms, df)


if __name__ == "__main__":
    report_only() if "--report" in sys.argv else main()
