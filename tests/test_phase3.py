"""Phase 3 checks: drug mapping, pharmacology features, two-stage and fusion models."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr import data, pharma  # noqa: E402
from adr.models import BlockAverage, CountStacked, Prior  # noqa: E402


def _toy(n=200, d=20, L=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Y = (X[:, :L] + 0.3 * rng.randn(n, L) > 0).astype(int)
    return X, Y


def test_drugs_map_to_stitch_ids():
    df, _ = data.load()
    t = pharma.drug_table(df)
    assert len(t) == len(df)
    assert t["Flat_CID"].notna().sum() >= 1400
    assert t["Flat_CID"].dropna().str.match(r"CID1\d{8}$").all()


@pytest.mark.skipif(not (pharma.CACHE / "sider_meddra_all_indications.tsv.gz").exists(),
                    reason="indications not downloaded (run the phase3 benchmark once)")
def test_indication_features():
    df, _ = data.load()
    X, names = pharma.indications(df)
    assert X.shape == (len(df), len(names)) and names[-1] == "has_indication"
    assert set(np.unique(X)) <= {0, 1}
    assert (X[:, :-1].sum(0) >= pharma.MIN_DRUGS_PER_INDICATION).all()
    assert np.array_equal(X[:, -1], X[:, :-1].any(1))


def test_count_stacked_shapes():
    X, Y = _toy()
    for use_x in (True, False):
        P = CountStacked(use_x=use_x, n_jobs=1).fit(X[:150], Y[:150]).predict_proba(X[150:])
        assert P.shape == (50, Y.shape[1]) and ((P >= 0) & (P <= 1)).all()


def test_block_average_uses_both_blocks():
    X, Y = _toy()
    m = BlockAverage(5, Prior, Prior).fit(X, Y)
    assert np.allclose(m.predict_proba(X[:2]), Y.mean(0))
