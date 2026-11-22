"""Per-perturbation metrics on pseudobulk profiles (1-D arrays over genes).

Definitions follow Miller et al. (shiftbioscience/Perturbation-Models-Outperform-
Baselines) for the error, weighted and rank families, and Cell-Eval for the DE-based
family, adapted to predictions that are mean profiles rather than cells.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def mse(pred, truth) -> float:
    return float(np.mean((pred - truth) ** 2))


def mae(pred, truth) -> float:
    return float(np.mean(np.abs(pred - truth)))


def wmse(pred, truth, weights) -> float:
    """Squared error with gene weights rescaled to sum to one."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w / w.sum() * (pred - truth) ** 2))


def deg_weights(scores) -> np.ndarray:
    """Miller et al. gene weights: min-max scaled |DE score|, squared; NaN scores get 0."""
    a = np.abs(np.asarray(scores, dtype=float))
    scaled = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
    return np.square(np.nan_to_num(scaled, nan=0.0))


def _correlation(test, x, y) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(test(x, y).statistic)


def pearson_delta(pred, truth, ref) -> float:
    """Pearson correlation of pred - ref with truth - ref (NaN if either is constant)."""
    return _correlation(stats.pearsonr, pred - ref, truth - ref)


def spearman_on(pred, truth, mask) -> float:
    """Spearman correlation restricted to the masked genes (NaN below three genes)."""
    if mask.sum() < 3:
        return float("nan")
    return _correlation(stats.spearmanr, pred[mask], truth[mask])


def de_pr_auc(pred_delta, is_de) -> float:
    """Average precision of |predicted change| as a score for the true DE genes."""
    if not is_de.any():
        return float("nan")
    return float(average_precision_score(is_de, np.abs(pred_delta)))


def direction_agreement(pred_delta, truth_delta, mask) -> float:
    """Fraction of masked genes whose predicted change has the true sign."""
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(pred_delta[mask]) == np.sign(truth_delta[mask])))


def top_k_overlap(pred_delta, true_top) -> float:
    """Share of the k true top genes found among the k largest |predicted changes|."""
    k = int(true_top.sum())
    if k == 0:
        return float("nan")
    predicted_top = np.argsort(-np.abs(pred_delta), kind="stable")[:k]
    return float(true_top[predicted_top].mean())


def nir(pred, truth) -> np.ndarray:
    """Normalized inverse rank, one value per row (perturbation).

    The fraction of other perturbations whose true profile is strictly farther
    (Euclidean) from this prediction than its own true profile is.
    """
    dist = cdist(pred, truth)
    n = len(dist)
    if n < 2:
        return np.full(n, np.nan)
    beats = np.diag(dist)[:, None] < dist
    np.fill_diagonal(beats, False)
    return beats.sum(axis=1) / (n - 1)
