"""Cross-validated evaluation with per-ADR ROC-AUC and PR-AUC.

Rules that the 2023 notebook broke and this module enforces:
  * folds are fixed before anything is fitted; every model is fitted on train folds only;
  * no resampling outside a fold (imbalance is handled by class weights inside the model);
  * metrics are threshold-free (ROC-AUC, PR-AUC), reported per ADR and averaged.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def cross_validate(make_model, X, Y, folds):
    """Returns (roc, pr, prevalence): arrays of shape (n_folds, n_labels), NaN where undefined."""
    k, L = len(folds), Y.shape[1]
    roc = np.full((k, L), np.nan)
    pr = np.full((k, L), np.nan)
    prev = np.full((k, L), np.nan)
    all_idx = np.arange(len(Y))
    for i, test in enumerate(folds):
        train = np.setdiff1d(all_idx, test)
        model = make_model().fit(X[train], Y[train])
        P = model.predict_proba(X[test])
        for j in range(L):
            y = Y[test, j]
            if y.min() == y.max():  # AUC undefined when the test fold has one class
                continue
            roc[i, j] = roc_auc_score(y, P[:, j])
            pr[i, j] = average_precision_score(y, P[:, j])
            prev[i, j] = y.mean()
    return roc, pr, prev
