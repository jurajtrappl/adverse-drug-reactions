"""Phase 3 analyses of the problem itself -> results/ANALYSIS.md.

1. Label structure: how much of the 27-label matrix is one "ADR count" factor?
2. Which ADR categories move together (label clusters)?
3. Can the ADR count be predicted from structure or from pharmacology?
4. Near-identical molecules with different labels: what are they?
5. Learning curves: would more drugs help?

Usage: python scripts/analyze.py   (after scripts/run_benchmark.py has cached the features)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from adr import data, features, pharma, splits  # noqa: E402
from adr.models import rf  # noqa: E402
from run_benchmark import featurize  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "results"
SEEDS = (0, 1, 2)
N_CLUSTERS = 6


def mean_auc(Y, P):
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])
                    if Y[:, j].min() != Y[:, j].max()])


def label_structure(Y, labels, out):
    count = Y.sum(1)
    pca = PCA().fit(Y)
    pc1 = pca.transform(Y)[:, 0]
    q = np.percentile(count, [0, 25, 50, 75, 100]).astype(int)
    oracle = np.mean([roc_auc_score(Y[:, j], count - Y[:, j]) for j in range(Y.shape[1])])
    ev = pca.explained_variance_ratio_
    out += ["## 1. One factor dominates the labels", "",
            f"A drug lists {q[2]} of the 27 ADR categories on median (quartiles {q[1]}-{q[3]}, "
            f"range {q[0]}-{q[4]}).", "",
            "| Principal component | 1 | 2 | 3 | 4 | 5 |", "| --- | --- | --- | --- | --- | --- |",
            "| Share of label variance | " + " | ".join(f"{v:.0%}" for v in ev[:5]) + " |", "",
            f"PC1 correlates with the drug's ADR count at |r| = {abs(np.corrcoef(pc1, count)[0, 1]):.3f}. "
            f"Knowing only how many *other* categories a drug lists predicts each category at mean "
            f"ROC-AUC **{oracle:.3f}**, far above any structure model (~0.65). The count behaves "
            "like a 'how long is the package insert' factor.", ""]


def label_clusters(Y, labels, out):
    C = np.corrcoef(Y.T)
    Z = linkage(squareform(1 - C, checks=False), "average")
    cl = fcluster(Z, N_CLUSTERS, "maxclust")
    prev = Y.mean(0)
    out += ["## 2. ADR categories that move together", "",
            f"Average-linkage clustering on 1 - correlation, cut into {N_CLUSTERS} groups. "
            "Mean within-group correlation in brackets.", "",
            "| Group | Categories (share of drugs listing it) | Within r |", "| --- | --- | --- |"]
    groups = sorted(set(cl), key=lambda g: -prev[cl == g].mean())
    for k, g in enumerate(groups, 1):
        idx = np.where(cl == g)[0]
        sub = C[np.ix_(idx, idx)]
        within = sub[np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else float("nan")
        names = ", ".join(f"{labels[i]} ({prev[i]:.0%})" for i in idx[np.argsort(-prev[idx])])
        out.append(f"| {k} | {names} | {within:.2f} |" if len(idx) > 1 else f"| {k} | {names} | — |")
    out.append("")


def count_regression(df, Y, out):
    count = Y.sum(1)
    rows = []
    for fname in ("morgan+desc", "atc+ind", "morgan+desc+atc+ind"):
        X = featurize(fname, df)
        rho, r2 = [], []
        for seed in SEEDS:
            folds = splits.load_folds(ROOT / "data" / "splits" / f"scaffold_seed{seed}.csv")
            pred = np.zeros(len(Y))
            for test in folds:
                train = np.setdiff1d(np.arange(len(Y)), test)
                m = RandomForestRegressor(300, min_samples_leaf=3, max_features=0.3, n_jobs=-1,
                                          random_state=seed).fit(X[train], count[train])
                pred[test] = m.predict(X[test])
            rho.append(spearmanr(pred, count).correlation)
            r2.append(r2_score(count, pred))
        rows.append((fname, np.mean(rho), np.mean(r2)))
        print("count regression", rows[-1], flush=True)
    out += ["## 3. Predicting the ADR count", "",
            "Random-forest regression of the number of listed ADR categories (0-27), scaffold "
            "split, 3 seeds.", "",
            "| Features | Spearman ρ | R² |", "| --- | --- | --- |"]
    out += [f"| {f} | {rho:.3f} | {r2:.3f} |" for f, rho, r2 in rows]
    out.append("")


def near_twins(df, Y, labels, out, min_sim=0.8, show=12):
    fps = features.morgan_bitvects(list(df["smiles"]))
    names = pharma.drug_table(df)["Title"].fillna("(no name)")
    rows = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, s in enumerate(sims):
            j = i + 1 + k
            if s >= min_sim and df["group"].iat[i] != df["group"].iat[j]:
                a, b = Y[i], Y[j]
                jac = (a & b).sum() / max((a | b).sum(), 1)
                rows.append((s, jac, i, j))
    t = pd.DataFrame(rows, columns=["tanimoto", "jaccard", "i", "j"])
    out += ["## 4. Near-identical molecules, different labels", "",
            f"{len(t)} pairs of *different* molecules have Morgan Tanimoto ≥ {min_sim}. Their ADR "
            f"label sets overlap by Jaccard {t.jaccard.mean():.2f} on average (median "
            f"{t.jaccard.median():.2f}); {(t.jaccard < 0.5).mean():.0%} of pairs share less than "
            "half their labels. Pairs with the least overlap:", "",
            "| Drug A | ADRs | Drug B | ADRs | Tanimoto | Label Jaccard |",
            "| --- | --- | --- | --- | --- | --- |"]
    for _, r in t.sort_values(["jaccard", "tanimoto"], ascending=[True, False]).head(show).iterrows():
        i, j = int(r.i), int(r.j)
        out.append(f"| {names.iat[i][:40]} | {Y[i].sum()} | {names.iat[j][:40]} | {Y[j].sum()} | "
                   f"{r.tanimoto:.2f} | {r.jaccard:.2f} |")
    out += ["", "Names are PubChem titles. The mismatches are mostly different *products* of the "
            "same chemistry: a nasal spray vs a skin cream (Beconase AQ vs betamethasone "
            "dipropionate), a combination eye drop vs the plain steroid (Prednefrin vs Pred "
            "Forte), a supplement vs a prescription drug (eicosapentaenoic acid vs Epanova). "
            "Route, dose and label length differ, and the structure can't see any of that.", ""]


def learning_curves(df, Y, out, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), seed=0):
    folds = splits.load_folds(ROOT / "data" / "splits" / f"scaffold_seed{seed}.csv")
    res = {}
    for fname in ("morgan+desc", "atc+ind"):
        X = featurize(fname, df)
        for frac in fractions:
            aucs = []
            for test in folds:
                train = np.setdiff1d(np.arange(len(Y)), test)
                rng = np.random.RandomState(seed)
                sub = rng.choice(train, int(round(frac * len(train))), replace=False)
                P = rf(seed=seed).fit(X[sub], Y[sub]).predict_proba(X[test])
                aucs.append(mean_auc(Y[test], P))
            res[(fname, frac)] = np.mean(aucs)
            print("learning curve", fname, frac, round(res[(fname, frac)], 3), flush=True)
    n_train = int(round(len(Y) * 0.8))
    out += ["## 6. Learning curves", "",
            f"Random forest, scaffold split (seed {seed}), trained on a random share of each "
            f"training fold (100% ≈ {n_train} drugs).", "",
            "| Training drugs | " + " | ".join(f"{int(f * n_train)}" for f in fractions) + " |",
            "| --- | " + " | ".join("---" for _ in fractions) + " |"]
    for fname in ("morgan+desc", "atc+ind"):
        out.append(f"| {fname} | " + " | ".join(f"{res[(fname, f)]:.3f}" for f in fractions) + " |")
    gain = res[("morgan+desc", 1.0)] - res[("morgan+desc", 0.4)]
    out += ["", f"Both curves still rise, slowly: structure gains {gain:+.3f} from "
            f"{int(0.4 * n_train)} to {n_train} drugs. More drugs would help a little; nothing "
            "suggests a data size at which structure alone would catch up with pharmacology.", ""]


def documentation_flags(df, Y, out):
    """How far do three 'how well documented is this drug' numbers go on their own?"""
    from adr.evaluate import cross_validate
    from adr.models import logreg
    ind, _ = pharma.indications(df)
    atc, _ = pharma.atc(df)
    X = np.column_stack([ind[:, -1], atc[:, -1], ind[:, :-1].sum(1)])
    rocs = []
    for seed in SEEDS:
        folds = splits.load_folds(ROOT / "data" / "splits" / f"scaffold_seed{seed}.csv")
        roc, _, _ = cross_validate(lambda: logreg(seed=seed), X, Y, folds)
        rocs.append(np.nanmean(roc))
    out += ["## 5. Documentation depth is a feature", "",
            "Three numbers with no chemistry and no drug-specific meaning (does the drug have "
            "parsed indications, was it matched to an ATC code, how many indications) give "
            f"mean ROC-AUC **{np.mean(rocs):.3f}** (logistic regression, scaffold split, 3 seeds), "
            "as good as the best structure model. Part of what the pharmacology features "
            "add is this 'well-documented drug' signal, which is the same factor as the ADR "
            "count in section 1.", ""]


def main():
    df, labels = data.load()
    Y = df[labels].to_numpy()
    cov = pharma.coverage(df)
    out = ["# Phase 3 analysis: understanding the problem", "",
           f"{len(df)} drugs, {len(labels)} ADR categories. Pharmacology coverage: "
           f"{cov['matched_cid']} drugs matched to a PubChem/STITCH id, {cov['with_indications']} "
           f"have SIDER indications ({cov['indication_terms']} terms used), {cov['with_atc']} "
           "matched to a WHO ATC code by name.", ""]
    label_structure(Y, labels, out)
    label_clusters(Y, labels, out)
    count_regression(df, Y, out)
    near_twins(df, Y, labels, out)
    documentation_flags(df, Y, out)
    learning_curves(df, Y, out)
    RES.mkdir(exist_ok=True)
    (RES / "ANALYSIS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
