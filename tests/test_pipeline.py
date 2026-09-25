"""Guards against the exact mistakes found in the 2023 audit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adr import data, splits  # noqa: E402
from adr.evaluate import cross_validate  # noqa: E402
from adr.models import Prior  # noqa: E402


@pytest.fixture(scope="module")
def sider():
    return data.load()


def test_counter_ions_removed_combinations_kept():
    assert data.clean_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert data.clean_smiles("C1C(O1)CCl.C(CNCCNCCNCCN)N") == "NCCNCCNCCNCCN"
    # etonogestrel + ethinylestradiol: both are active, both stay
    combo = ("CCC12CC(=C)C3C(C1CC[C@]2(C#C)O)CCC4=CC(=O)CCC34."
             "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@]2(C#C)O)CCC4=C3C=CC(=C4)O")
    assert "." in data.clean_smiles(combo)


def test_labels_and_shape(sider):
    df, labels = sider
    assert len(df) == 1427 and len(labels) == 27
    assert set(np.unique(df[labels].to_numpy())) == {0, 1}


@pytest.mark.parametrize("split", ["scaffold", "random"])
@pytest.mark.parametrize("seed", [0, 1])
def test_folds_partition_and_no_leakage(sider, split, seed):
    df, _ = sider
    folds = splits.SPLITTERS[split](df["scaffold"], df["group"], k=5, seed=seed)
    allidx = np.concatenate(folds)
    assert len(allidx) == len(df) == len(set(allidx))  # every drug exactly once
    assert min(map(len, folds)) > 0.15 * len(df)  # roughly balanced folds
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        # the same parent molecule never sits on both sides (e.g. two salts of one drug)
        assert not set(df["group"].iloc[test]) & set(df["group"].iloc[train])
        if split == "scaffold":
            keys = np.array(splits.scaffold_keys(df["scaffold"], df["group"]))
            assert not set(keys[test]) & set(keys[train])
            ring = df["scaffold"].to_numpy() != ""  # real ring scaffolds never cross folds
            assert not set(df["scaffold"].iloc[test][ring[test]]) & set(df["scaffold"].iloc[train][ring[train]])


def test_scaffold_folds_deterministic(sider):
    df, _ = sider
    a = splits.scaffold_folds(df["scaffold"], df["group"], seed=3)
    b = splits.scaffold_folds(df["scaffold"], df["group"], seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_prior_baseline_is_chance(sider):
    df, labels = sider
    Y = df[labels].to_numpy()
    folds = splits.random_folds(df["scaffold"], df["group"], seed=0)
    roc, _, _ = cross_validate(Prior, np.zeros((len(Y), 1)), Y, folds)
    assert np.allclose(np.nanmean(roc), 0.5)


def test_scaffold_folds_spread_acyclic_and_vary_by_seed(sider):
    df, _ = sider
    acyclic = (df["scaffold"] == "").to_numpy()
    placements = []
    for seed in (0, 1, 2):
        folds = splits.scaffold_folds(df["scaffold"], df["group"], seed=seed)
        per_fold = [int(acyclic[f].sum()) for f in folds]
        assert max(per_fold) < 0.5 * acyclic.sum()  # not all ring-free drugs in one fold
        assert max(map(len, folds)) - min(map(len, folds)) <= 0.05 * len(df)
        benzene = (df["scaffold"] == "c1ccccc1").to_numpy()
        placements.append(next(i for i, f in enumerate(folds) if benzene[f].any()))
    assert len(set(placements)) > 1  # the biggest scaffold set doesn't sit in the same fold every seed

