"""Cross-validation folds. Both splitters keep identical molecules (same `group`) together."""
import numpy as np
import pandas as pd


def scaffold_folds(scaffolds, groups, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Murcko-scaffold k-fold: a scaffold never appears in two folds.

    Scaffold sets are shuffled (seed), then placed largest-first into the currently
    smallest fold, which keeps fold sizes balanced. `groups` is accepted for API symmetry;
    identical molecules share a scaffold, so they are already kept together.
    """
    buckets: dict[str, list[int]] = {}
    for i, s in enumerate(scaffolds):
        buckets.setdefault(s, []).append(i)
    sets = list(buckets.values())
    np.random.RandomState(seed).shuffle(sets)
    sets.sort(key=len, reverse=True)  # stable sort: ties keep the shuffled order
    folds: list[list[int]] = [[] for _ in range(k)]
    for members in sets:
        min(folds, key=len).extend(members)
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
