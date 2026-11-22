"""Preprocessing shared by the dataset scripts."""

import numpy as np
import pandas as pd
from scipy import sparse


def log_normalize(X, library, target_sum: float = 1e4) -> sparse.csr_matrix:
    """log1p(counts / library * target_sum) per cell (rows), in place when X is already CSR.

    `library` is each cell's total count over all genes. It can exceed the row sums
    of X when X holds only some of the genes.
    """
    X = sparse.csr_matrix(X)
    scale = (target_sum / np.asarray(library, dtype=np.float64)).astype(X.dtype)
    X.data *= np.repeat(scale, np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X


def split_conditions(conditions, control: str, seed: int, test_fraction: float) -> pd.DataFrame:
    """Hold out a random share of the combination conditions (names joined with "+").

    Every single perturbation stays in training. Returns one row per non-control
    condition with columns condition, kind ("single" or "combination") and split
    ("train" or "test").
    """
    names = sorted(set(conditions) - {control})
    combinations = [c for c in names if "+" in c]
    rng = np.random.default_rng(seed)
    test = set(rng.choice(combinations, size=round(len(combinations) * test_fraction), replace=False))
    return pd.DataFrame(
        {
            "condition": names,
            "kind": ["combination" if "+" in c else "single" for c in names],
            "split": ["test" if c in test else "train" for c in names],
        }
    )
