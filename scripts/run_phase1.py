"""Phase 1 benchmark: every model x split x seed, results written to results/.

Usage (from the repo root):
    python scripts/run_phase1.py                      # full run
    python scripts/run_phase1.py --models prior morgan+rf --seeds 0

Finished (split, seed, model) runs are saved after each model and skipped on the next
invocation, so an interrupted run can simply be restarted. Use --fresh to start over.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adr import data, features, splits  # noqa: E402
from adr.evaluate import cross_validate  # noqa: E402
from adr.models import EXPERIMENTS  # noqa: E402

RESULTS = ROOT / "results"
SPLIT_DIR = ROOT / "data" / "splits"


CACHE = ROOT / ".cache"


def featurize(name, df):
    """Featurize once and cache to .cache/ (RDKit descriptors take a while)."""
    if name is None:
        return np.zeros((len(df), 1))
    path = CACHE / f"features_{name.replace('+', '_')}.npy"
    if path.exists():
        return np.load(path)
    x = features.FEATURIZERS[name](list(df["smiles"]))
    CACHE.mkdir(exist_ok=True)
    np.save(path, x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--tag", default="phase1")
    ap.add_argument("--fresh", action="store_true", help="ignore previously saved runs")
    args = ap.parse_args()
    runs_path = RESULTS / f"{args.tag}_runs.csv"
    adr_path = RESULTS / f"{args.tag}_per_adr.csv"

    df, labels = data.load()
    Y = df[labels].to_numpy()
    print(f"{len(df)} drugs, {len(labels)} ADRs, {df['scaffold'].nunique()} scaffolds", flush=True)

    feats = {}
    rows, per_adr = [], []
    if runs_path.exists() and not args.fresh:
        rows = pd.read_csv(runs_path).to_dict("records")
        per_adr = pd.read_csv(adr_path).to_dict("records")
    done = {(r["split"], r["seed"], r["model"]) for r in rows}
    RESULTS.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for seed in args.seeds:
            folds = splits.SPLITTERS[split](df["scaffold"], df["group"], k=5, seed=seed)
            splits.save_folds(folds, SPLIT_DIR / f"{split}_seed{seed}.csv")
            for name in args.models:
                if (split, seed, name) in done:
                    continue
                feat_name, factory = EXPERIMENTS[name]
                if feat_name not in feats:
                    feats[feat_name] = featurize(feat_name, df)
                t0 = time.time()
                roc, pr, prev = cross_validate(
                    lambda: factory(n_jobs=args.jobs, seed=seed), feats[feat_name], Y, folds)
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

    runs = pd.DataFrame(rows)
    runs = runs[runs["split"].isin(args.splits) & runs["seed"].isin(args.seeds)
                & runs["model"].isin(args.models)]
    summary = (runs.groupby(["split", "model"], sort=False)
               .agg(roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                    pr_auc_lift=("pr_auc_lift", "mean"), seeds=("seed", "count"))
               .reset_index().round(3))
    summary.to_csv(RESULTS / f"{args.tag}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
