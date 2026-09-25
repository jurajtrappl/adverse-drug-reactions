"""Benchmark: every model x split x seed of an experiment set, results written to results/.

Usage (from the repo root):
    python scripts/run_benchmark.py --set phase1        # classic ML, scaffold + random split
    python scripts/run_benchmark.py --set phase2        # neural + pretrained, scaffold split
    python scripts/run_benchmark.py --set phase3        # pharmacology + label-count models
    python scripts/run_benchmark.py --set phase2 --models chemprop --seeds 0

Resumable: finished (split, seed, model) runs are saved after each model and each fold's
predictions are cached in .cache/preds/, so an interrupted run can simply be restarted.
--fresh recomputes the selected (split, seed, model) runs, ignoring their saved rows and
cached predictions; other rows in the results files are kept. Caches are keyed by a hash of
the adr/ package and the RDKit version, so editing the code never reuses stale features or
predictions. Folds come from data/splits/ so every model sees the same folds.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adr import data, features, pharma, splits  # noqa: E402
from adr.evaluate import cross_validate  # noqa: E402
from adr.models import EXPERIMENTS  # noqa: E402

RESULTS = ROOT / "results"
SPLIT_DIR = ROOT / "data" / "splits"
CACHE = ROOT / ".cache"
PHARMA = {"atc+ind": ["atc", "ind"], "morgan+desc+atc+ind": ["morgan+desc", "atc", "ind"]}


def code_version() -> str:
    """Short hash of the adr/ sources, the cleaned data and the RDKit version (cache key)."""
    import rdkit
    h = hashlib.sha1(rdkit.__version__.encode())
    for f in sorted((ROOT / "adr").glob("*.py")) + [ROOT / "data" / "sider_clean.csv"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


VERSION = code_version()


def featurize(name, df):
    """Featurize once and cache to .cache/ (RDKit descriptors take a while)."""
    if name is None:
        return np.zeros((len(df), 1))
    if name == "smiles":  # graph models featurize internally
        return df["smiles"].to_numpy(dtype=object)
    if name == "smiles+desc":  # graph model + RDKit descriptors fed to its output layers
        out = np.empty((len(df), 2), dtype=object)
        out[:, 0] = df["smiles"].to_numpy()
        out[:, 1] = list(featurize("desc", df))
        return out
    path = CACHE / VERSION / f"features_{name.replace('+', '_')}.npy"
    if path.exists():
        return np.load(path)
    if name in PHARMA:
        x = np.hstack([featurize(part, df) for part in PHARMA[name]])
    elif name in ("atc", "ind"):
        x = (pharma.atc if name == "atc" else pharma.indications)(df)[0]
    else:
        x = features.FEATURIZERS[name](list(df["smiles"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(EXPERIMENTS), default="phase1")
    ap.add_argument("--models", nargs="+", help="default: every model in the set")
    ap.add_argument("--splits", nargs="+", help="default: scaffold+random (phase1), scaffold (phase2)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--tag", help="results file prefix, default: the set name")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute the selected runs (ignore their saved rows and cached predictions)")
    args = ap.parse_args()
    experiments = EXPERIMENTS[args.set]
    args.models = args.models or list(experiments)
    args.splits = args.splits or (["scaffold", "random"] if args.set == "phase1" else ["scaffold"])
    args.tag = args.tag or args.set
    runs_path = RESULTS / f"{args.tag}_runs.csv"
    adr_path = RESULTS / f"{args.tag}_per_adr.csv"

    df, labels = data.load()
    Y = df[labels].to_numpy()
    print(f"{len(df)} drugs, {len(labels)} ADRs, {df['scaffold'].nunique()} scaffolds", flush=True)

    feats = {}
    rows, per_adr = [], []
    if runs_path.exists():
        rows = pd.read_csv(runs_path).to_dict("records")
        per_adr = pd.read_csv(adr_path).to_dict("records")
    selected = {(sp, sd, m) for sp in args.splits for sd in args.seeds for m in args.models}
    if args.fresh:  # drop only the runs being recomputed
        rows = [r for r in rows if (r["split"], r["seed"], r["model"]) not in selected]
        per_adr = [r for r in per_adr if (r["split"], r["seed"], r["model"]) not in selected]
    done = {(r["split"], r["seed"], r["model"]) for r in rows}
    RESULTS.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for seed in args.seeds:
            fold_file = SPLIT_DIR / f"{split}_seed{seed}.csv"
            if fold_file.exists():
                folds = splits.load_folds(fold_file)
            else:
                folds = splits.SPLITTERS[split](df["scaffold"], df["group"], k=5, seed=seed)
                splits.save_folds(folds, fold_file)
            for name in args.models:
                if (split, seed, name) in done:
                    continue
                feat_name, factory = experiments[name]
                if feat_name not in feats:
                    feats[feat_name] = featurize(feat_name, df)
                t0 = time.time()
                safe = name.replace("+", "_").replace(" ", "_")
                prefix = CACHE / "preds" / VERSION / f"{args.tag}_{split}_s{seed}_{safe}"
                if args.fresh:
                    for old in prefix.parent.glob(prefix.name + "_fold*.npy"):
                        old.unlink()
                roc, pr, prev = cross_validate(
                    lambda: factory(n_jobs=args.jobs, seed=seed), feats[feat_name], Y, folds,
                    cache_prefix=prefix)
                roc_adr, pr_adr = np.nanmean(roc, 0), np.nanmean(pr, 0)
                lift_adr = pr_adr - np.nanmean(prev, 0)
                rows.append(dict(split=split, seed=seed, model=name,
                                 roc_auc=roc_adr.mean(), pr_auc_lift=lift_adr.mean(),
                                 seconds=round(time.time() - t0, 1)))
                per_adr += [dict(split=split, seed=seed, model=name, adr=a, roc_auc=r, pr_auc=p,
                                 prevalence=Y[:, j].mean())
                            for j, (a, r, p) in enumerate(zip(labels, roc_adr, pr_adr))]
                print(f"{split:8s} seed={seed} {name:22s} ROC-AUC={roc_adr.mean():.3f} "
                      f"({rows[-1]['seconds']}s)", flush=True)
                pd.DataFrame(rows).to_csv(runs_path, index=False)
                pd.DataFrame(per_adr).to_csv(adr_path, index=False)

    runs = pd.DataFrame(rows)  # every saved run of this set, not only this invocation's
    summary = (runs.groupby(["split", "model"], sort=False)
               .agg(roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                    pr_auc_lift=("pr_auc_lift", "mean"), seeds=("seed", "count"))
               .reset_index().round(3))
    summary.to_csv(RESULTS / f"{args.tag}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
