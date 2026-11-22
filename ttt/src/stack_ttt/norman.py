"""Norman et al. (2019) cells on STACK's gene vocabulary, and the TTT cell pools."""

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from stack_robustness.baselines import assign_halves


@dataclass
class Cells:
    counts: sparse.csr_matrix  # cells x STACK vocabulary, raw counts (0 where the data lack a gene)
    library: np.ndarray  # each cell's total counts over all genes (scPerturb's `ncounts`)
    condition: np.ndarray  # per cell; doubles are written "A+B"
    genes: list  # STACK's vocabulary, in column order

    def rows(self, condition: str) -> np.ndarray:
        return np.flatnonzero(self.condition == condition)


def load_norman(raw_path, vocab, conditions=None, aliases=None) -> Cells:
    """Raw counts of the requested conditions (all if None), columns in vocabulary order.

    Genes are matched case-insensitively, as STACK's own data loaders do, after renaming
    them with `aliases` (data symbol -> vocabulary symbol; see
    stack_robustness.genes.vocabulary_aliases).
    """
    backed = ad.read_h5ad(raw_path, backed="r")
    condition = backed.obs["perturbation"].astype(str).str.replace("_", "+", regex=False).to_numpy()
    keep = np.ones(len(condition), bool) if conditions is None else np.isin(condition, list(conditions))
    position = {g.upper(): j for j, g in enumerate(vocab)}
    aliases = aliases or {}
    names = [aliases.get(str(g), str(g)).upper() for g in backed.var_names]
    data_cols = np.array([i for i, g in enumerate(names) if g in position])
    vocab_cols = np.array([position[names[i]] for i in data_cols])
    X = sparse.csc_matrix(backed.X[:, data_cols])  # stored column-major: reads only these genes
    library = backed.obs["ncounts"].to_numpy(dtype=float)[keep]
    backed.file.close()
    X = X.tocsr()[np.flatnonzero(keep)].tocoo()
    counts = sparse.csr_matrix((X.data, (X.row, vocab_cols[X.col])), shape=(X.shape[0], len(vocab)), dtype=np.float32)
    return Cells(counts=counts, library=library, condition=condition[keep], genes=list(vocab))


def single_pools(condition: np.ndarray, singles, seed: int) -> dict:
    """Split each single perturbation's cells into disjoint "adapt" and "infer" rows.

    Adaptation tasks draw single-perturbation cells from "adapt"; predictions for a held-out
    double use "infer", so adaptation can never copy the cells it is later asked about.
    """
    labels = pd.Series(condition)
    labels = labels[labels.isin(list(singles))]
    halves = assign_halves(labels, seed=seed)
    pools = {}
    for single in singles:
        own = labels == single
        pools[(single, "adapt")] = halves.index[own & (halves == "first")].to_numpy()
        pools[(single, "infer")] = halves.index[own & (halves == "second")].to_numpy()
    return pools
