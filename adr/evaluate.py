"""Cross-validated evaluation with per-ADR ROC-AUC and PR-AUC.

Rules that the 2023 notebook broke and this module enforces:
  * folds are fixed before anything is fitted; every model is fitted on train folds only;
  * no resampling outside a fold (imbalance is handled by class weights inside the model);
  * metrics are threshold-free (ROC-AUC, PR-AUC), reported per ADR and averaged.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def cross_validate(make_model, X, Y, folds, cache_prefix=None):
    """Returns (roc, pr, prevalence): arrays of shape (n_folds, n_labels), NaN where undefined.

    With `cache_prefix` (a path without extension), each fold's test-set predictions are
    saved as <prefix>_fold<i>.npy and reused, so slow models can resume fold by fold.
    """
    k, L = len(folds), Y.shape[1]
    roc = np.full((k, L), np.nan)
    pr = np.full((k, L), np.nan)
    prev = np.full((k, L), np.nan)
    all_idx = np.arange(len(Y))
    for i, test in enumerate(folds):
        cached = Path(f"{cache_prefix}_fold{i}.npy") if cache_prefix else None
        if cached is not None and cached.exists():
            P = np.load(cached)
        else:
            train = np.setdiff1d(all_idx, test)
            model = make_model().fit(X[train], Y[train])
            P = model.predict_proba(X[test])
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, P)
        for j in range(L):
            y = Y[test, j]
            if y.min() == y.max():  # AUC undefined when the test fold has one class
                continue
            roc[i, j] = roc_auc_score(y, P[:, j])
            pr[i, j] = average_precision_score(y, P[:, j])
            prev[i, j] = y.mean()
    return roc, pr, prev
