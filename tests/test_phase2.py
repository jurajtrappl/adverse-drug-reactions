"""Phase 2 checks: fold files round-trip, neural models learn and keep the interface."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr import data, splits  # noqa: E402
from adr.models import Average, Prior  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_saved_folds_match_regenerated():
    df, _ = data.load()
    for seed in (0, 1, 2):
        saved = splits.load_folds(ROOT / "data" / "splits" / f"scaffold_seed{seed}.csv")
        fresh = splits.scaffold_folds(df["scaffold"], df["group"], seed=seed)
        assert all(np.array_equal(a, b) for a, b in zip(saved, fresh))


def _toy(n=300, d=40, L=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = (X[:, :L] + 0.3 * rng.randn(n, L) > 0).astype(int)  # label j depends on feature j
    return X, Y


def test_multitask_mlp_learns():
    pytest.importorskip("torch")
    from adr.nn import MultiTaskMLP, mean_auc
    X, Y = _toy()
    m = MultiTaskMLP(hidden=(32,), max_epochs=60, seed=0).fit(X[:200], Y[:200])
    P = m.predict_proba(X[200:])
    assert P.shape == (100, 5) and ((P >= 0) & (P <= 1)).all()
    assert mean_auc(Y[200:], P) > 0.7  # chance = 0.5


def test_average_ensemble():
    X, Y = _toy()
    P = Average(Prior(), Prior()).fit(X, Y).predict_proba(X[:3])
    assert np.allclose(P, Y.mean(0))
