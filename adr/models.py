"""Model zoo. Every model maps X (n x d) and Y (n x 27, binary) to probabilities (n x 27)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PerLabel:
    """One independent binary classifier per ADR (what the 2023 notebook did, minus leakage)."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, Y):
        self.models_ = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if y.min() == y.max():  # label constant in this training fold
                self.models_.append(float(y[0]))
            else:
                self.models_.append(clone(self.base).fit(X, y))
        return self

    def predict_proba(self, X):
        cols = [np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                for m in self.models_]
        return np.column_stack(cols)


class Prior:
    """Predicts each ADR's training-set frequency for every drug. ROC-AUC = 0.5 by construction."""

    def fit(self, X, Y):
        self.p_ = Y.mean(axis=0)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


class TanimotoKNN:
    """'Copy the labels of the k most similar training drugs' (Jaccard on bits = 1 - Tanimoto)."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, Y):
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric="jaccard", algorithm="brute")
        self.nn_.fit(X.astype(bool))
        self.Y_ = Y
        return self

    def predict_proba(self, X):
        _, idx = self.nn_.kneighbors(X.astype(bool))
        return self.Y_[idx].mean(axis=1)


class MultiOutputRF:
    """One forest shared by all 27 ADRs: every tree predicts the whole label vector."""

    def __init__(self, n_jobs: int = -1, seed: int = 0):
        self.rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         class_weight="balanced_subsample",
                                         n_jobs=n_jobs, random_state=seed)

    def fit(self, X, Y):
        self.rf.fit(X, Y)
        return self

    def predict_proba(self, X):
        return np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(X))
                                for p in self.rf.predict_proba(X)])


def rf(n_jobs=-1, seed=0):
    return PerLabel(RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                           class_weight="balanced_subsample",
                                           n_jobs=n_jobs, random_state=seed))


def logreg(n_jobs=-1, seed=0):
    return PerLabel(make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.05, class_weight="balanced",
                                                     max_iter=5000)))


def lgbm(n_jobs=-1, seed=0):
    return PerLabel(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.3, class_weight="balanced",
                                   n_jobs=n_jobs, random_state=seed, verbose=-1))


class Average:
    """Ensemble: mean of the members' predicted probabilities."""

    def __init__(self, *members):
        self.members = members

    def fit(self, X, Y):
        for m in self.members:
            m.fit(X, Y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def mlp(n_jobs=-1, seed=0):
    from adr.nn import MultiTaskMLP  # torch is only needed for Phase 2
    return MultiTaskMLP(seed=seed, n_jobs=n_jobs)


def chemprop(n_jobs=-1, seed=0):
    from adr.nn import Chemprop
    return Chemprop(seed=seed, n_jobs=n_jobs)


# name -> (featurizer name, factory)
PHASE1 = {
    "prior":                (None,          lambda **kw: Prior()),
    "size+logreg":          ("size",        logreg),
    "knn5 tanimoto":        ("morgan",      lambda **kw: TanimotoKNN(5)),
    "morgan+logreg":        ("morgan",      logreg),
    "morgan+rf":            ("morgan",      rf),
    "desc+rf":              ("desc",        rf),
    "morgan+desc+rf":       ("morgan+desc", rf),
    "morgan+desc+rf-multi": ("morgan+desc", lambda **kw: MultiOutputRF(**kw)),
    "morgan+desc+lgbm":     ("morgan+desc", lgbm),
}

PHASE2 = {
    "morgan+desc+mlp":      ("morgan+desc", mlp),
    "mol2vec+logreg":       ("mol2vec",     logreg),
    "mol2vec+rf":           ("mol2vec",     rf),
    "mol2vec+mlp":          ("mol2vec",     mlp),
    "mlp + rf ensemble":    ("morgan+desc", lambda **kw: Average(mlp(**kw), rf(**kw))),
    "chemprop":             ("smiles",      chemprop),
    "chemprop+desc":        ("smiles+desc", chemprop),
}

EXPERIMENTS = {"phase1": PHASE1, "phase2": PHASE2}
