"""Paired comparisons of two models across the same held-out perturbations."""

import numpy as np


def paired_summary(differences, n_boot: int = 10000, seed: int = 0) -> dict:
    """Median of per-perturbation differences, a percentile-bootstrap 95% interval for it
    (resampling perturbations), and how many differences are positive. NaNs are dropped."""
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "median": float(np.median(d)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_positive": int((d > 0).sum()),
    }
