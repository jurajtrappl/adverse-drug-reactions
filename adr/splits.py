"""Cross-validation folds. Both splitters keep identical molecules (same `group`) together."""
import numpy as np
import pandas as pd


def scaffold_keys(scaffolds, groups) -> list[str]:
    """Scaffold per drug; acyclic drugs (empty Murcko scaffold) get one key per molecule.

    Without this, all 156 ring-free drugs share the scaffold '' and land in one test fold,
    although they have nothing in common structurally.
    """
    return [s if s else f"acyclic:{g}" for s, g in zip(scaffolds, groups)]


def scaffold_folds(scaffolds, groups, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Murcko-scaffold k-fold: a scaffold never appears in two folds.

    Sets holding more than 5% of the drugs are placed first (so folds stay balanced), the
    rest in random order (seed). Each set goes to the currently smallest fold, ties broken at
    random, so the big sets don't sit in fold 0 for every seed. Identical molecules share a
    group and a scaffold, so they always stay together.
    """
    rng = np.random.RandomState(seed)
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(scaffold_keys(scaffolds, groups)):
        buckets.setdefault(s, []).append(i)
    sets = list(buckets.values())
    rng.shuffle(sets)
    big = [m for m in sets if len(m) > 0.05 * len(scaffolds)]
    small = [m for m in sets if len(m) <= 0.05 * len(scaffolds)]
    folds: list[list[int]] = [[] for _ in range(k)]
    for members in sorted(big, key=len, reverse=True) + small:
        sizes = np.array([len(f) for f in folds])
        target = rng.choice(np.flatnonzero(sizes == sizes.min()))
        folds[target].extend(members)
    return [np.sort(np.array(f)) for f in folds]


def random_folds(scaffolds, groups, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Random k-fold over molecule groups (salt forms of one drug stay together)."""
    groups = np.asarray(groups)
    uniq = np.random.RandomState(seed).permutation(np.unique(groups))
    fold_of_group = {g: i % k for i, g in enumerate(uniq)}
    fold = np.array([fold_of_group[g] for g in groups])
    return [np.where(fold == i)[0] for i in range(k)]


SPLITTERS = {"scaffold": scaffold_folds, "random": random_folds}


def save_folds(folds, path) -> None:
    fold = np.empty(sum(len(f) for f in folds), dtype=int)
    for i, f in enumerate(folds):
        fold[f] = i
    pd.DataFrame({"row": np.arange(len(fold)), "fold": fold}).to_csv(path, index=False)


def load_folds(path) -> list[np.ndarray]:
    fold = pd.read_csv(path).sort_values("row")["fold"].to_numpy()
    return [np.where(fold == i)[0] for i in range(fold.max() + 1)]
