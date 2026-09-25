"""Model zoo. Every model maps X (n x d) and Y (n x 27, binary) to probabilities (n x 27)."""
import numpy as np
from rdkit.Chem import Descriptors
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
    """One forest shared by all ADRs: every tree predicts the whole label vector.

    No class weights: for multi-output targets scikit-learn *multiplies* the per-output
    weights into one sample weight, which with many outputs gives absurd weights
    (27 outputs: -0.01 ROC-AUC; 579 outputs: below chance).
    """

    def __init__(self, n_jobs: int = -1, seed: int = 0):
        self.rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         n_jobs=n_jobs, random_state=seed)

    def fit(self, X, Y):
        self.rf.fit(X, Y)
        return self

    def predict_proba(self, X):
        # An output seen with a single class in training has a one-column probability array;
        # its prediction is that class (1.0 if the only class was 1), not 0.
        return np.column_stack([p[:, list(c).index(1)] if 1 in c else np.zeros(len(X))
                                for p, c in zip(self.rf.predict_proba(X), self.rf.classes_)])


def rf(n_jobs=-1, seed=0):
    return PerLabel(RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                           class_weight="balanced_subsample",
                                           n_jobs=n_jobs, random_state=seed))


def logreg(n_jobs=-1, seed=0):
    return PerLabel(make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.05, class_weight="balanced",
                                                     max_iter=5000)))


def lgbm(n_jobs=-1, seed=0):
    # Imported here: on macOS, loading LightGBM's and PyTorch's OpenMP runtimes in the same
    # process can abort with "OMP: Error #15"; Phase 2 (torch) never needs LightGBM.
    from lightgbm import LGBMClassifier
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


class CountStacked:
    """Two stages, exploiting the dominant label factor ("how many ADRs does this drug list").

    1. A random-forest regressor predicts the number of ADRs from X. Its training-set
       predictions are out-of-fold (inner 5-fold), so stage 2 sees realistic, noisy counts.
    2. Per ADR: a random forest on [X, predicted count] (use_x=True), or a logistic
       regression on the predicted count alone (use_x=False) to measure how much of the
       signal the count carries by itself.
    """

    def __init__(self, use_x=True, n_jobs=-1, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        self.use_x, self.n_jobs, self.seed = use_x, n_jobs, seed
        self.reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.3,
                                         n_jobs=n_jobs, random_state=seed)

    def _stack(self, X, c):
        return np.column_stack([X, c]) if self.use_x else c.reshape(-1, 1)

    def fit(self, X, Y):
        from sklearn.model_selection import KFold, cross_val_predict
        count = Y.sum(axis=1)
        oof = cross_val_predict(clone(self.reg), X, count,
                                cv=KFold(5, shuffle=True, random_state=self.seed))
        self.reg.fit(X, count)
        self.clf = (rf if self.use_x else logreg)(n_jobs=self.n_jobs, seed=self.seed)
        self.clf.fit(self._stack(X, oof), Y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._stack(X, self.reg.predict(X)))


class BlockAverage:
    """Late fusion: one model per column block, probabilities averaged.

    A single forest over [2,258 structure columns + ~370 pharmacology columns] mostly samples
    fingerprint bits at each split and drowns the pharmacology signal; separate forests don't.
    """

    def __init__(self, n_first, make_first, make_second):
        self.n_first, self.a, self.b = n_first, make_first(), make_second()

    def fit(self, X, Y):
        if not 0 < self.n_first < X.shape[1]:
            raise ValueError(f"block split at column {self.n_first} but X has {X.shape[1]} columns "
                             "(stale feature cache from another RDKit version? clear .cache/)")
        self.a.fit(X[:, :self.n_first], Y)
        self.b.fit(X[:, self.n_first:], Y)
        return self

    def predict_proba(self, X):
        return (self.a.predict_proba(X[:, :self.n_first])
                + self.b.predict_proba(X[:, self.n_first:])) / 2


def mlp(n_jobs=-1, seed=0):
    from adr.nn import MultiTaskMLP  # torch is only needed for Phase 2
    return MultiTaskMLP(seed=seed, n_jobs=n_jobs)


def chemprop(n_jobs=-1, seed=0):
    from adr.nn import Chemprop
    return Chemprop(seed=seed, n_jobs=n_jobs)


N_STRUCTURE = 2048 + len(Descriptors.descList)  # width of the morgan+desc block

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

PHASE3 = {
    "count-only":              ("morgan+desc",         lambda **kw: CountStacked(use_x=False, **kw)),
    "count-stacked rf":        ("morgan+desc",         lambda **kw: CountStacked(use_x=True, **kw)),
    "atc+rf":                  ("atc",                 rf),
    "ind+rf":                  ("ind",                 rf),
    "atc+ind+rf":              ("atc+ind",             rf),
    "morgan+desc+atc+ind+rf":  ("morgan+desc+atc+ind", rf),
    "structure rf + pharma rf": ("morgan+desc+atc+ind", lambda **kw: BlockAverage(
        N_STRUCTURE, lambda: rf(**kw), lambda: rf(**kw))),
}

EXPERIMENTS = {"phase1": PHASE1, "phase2": PHASE2, "phase3": PHASE3}
