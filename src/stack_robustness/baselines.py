"""Pseudobulk profiles and the shared baselines.

Every profile is a mean of log-normalized expression over cells, indexed by gene.
The positive and negative controls follow Miller et al. (shiftbioscience/
Perturbation-Models-Outperform-Baselines): each perturbation's cells are split in
half, the first half is the ground truth every model is scored against, and the
second half is the technical duplicate.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def pseudobulk(X, labels: pd.Series, genes) -> pd.DataFrame:
    """Mean expression of the cells in each group (rows of X follow `labels`)."""
    codes, groups = pd.factorize(pd.Series(labels).to_numpy(), sort=True)
    counts = np.bincount(codes, minlength=len(groups))
    indicator = sparse.csr_matrix(
        (1.0 / counts[codes], (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    means = indicator @ X
    if sparse.issparse(means):
        means = means.toarray()
    return pd.DataFrame(np.asarray(means), index=groups, columns=list(genes))


def assign_halves(labels: pd.Series, seed: int) -> pd.Series:
    """Randomly split each group's cells into a "first" and a "second" half.

    The first half gets floor(n/2) cells. Groups with fewer than two cells are left
    unassigned (NA), since they cannot provide both a truth and a duplicate.
    """
    rng = np.random.default_rng(seed)
    halves = pd.Series(pd.NA, index=labels.index, dtype="object")
    for _, cells in sorted(labels.groupby(labels, observed=True).groups.items()):
        if len(cells) < 2:
            continue
        shuffled = rng.permutation(np.asarray(cells))
        n_first = len(shuffled) // 2
        halves[shuffled[:n_first]] = "first"
        halves[shuffled[n_first:]] = "second"
    return halves


def mean_baseline(profiles: pd.DataFrame, train) -> pd.Series:
    """Average of the training perturbations' profiles, each weighted equally."""
    return profiles.loc[list(train)].mean(axis=0)


def interpolated_duplicate(dup: pd.Series, mean: pd.Series, pvals_adj: pd.Series) -> pd.Series:
    """Per gene, alpha * duplicate + (1 - alpha) * mean with alpha = 1 - adjusted p.

    Genes are matched by name. Genes without a p-value fall back to the mean
    baseline (alpha = 0).
    """
    alpha = (1.0 - pvals_adj.reindex(dup.index)).fillna(0.0)
    return alpha * dup + (1.0 - alpha) * mean.reindex(dup.index)


def additive(ctrl: pd.Series, singles: pd.DataFrame, parts) -> pd.Series:
    """Control plus the sum of each single perturbation's effect relative to control."""
    return ctrl + sum(singles.loc[part] - ctrl for part in parts)


def source_effect_copy(prompt_pert: pd.Series, prompt_ctrl: pd.Series, query_ctrl: pd.Series) -> pd.Series:
    """Transfer the prompt context's measured effect unchanged onto the query controls."""
    return query_ctrl + (prompt_pert - prompt_ctrl)
