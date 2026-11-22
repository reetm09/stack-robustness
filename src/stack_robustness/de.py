"""Differential expression via scanpy, reshaped to one gene-aligned table per group."""

import pandas as pd
import scanpy as sc

_KEY = "_stack_robustness_de"


def rank_genes(adata, groupby: str, reference: str = "rest", method: str = "t-test_overestim_var") -> dict:
    """Test every group against `reference` ("rest" or a group name).

    `adata.X` must be log-normalized. Returns {group: DataFrame indexed like
    adata.var_names with columns score, pval, pval_adj}; the reference group itself
    is left out.
    """
    sc.tl.rank_genes_groups(adata, groupby, reference=reference, method=method, key_added=_KEY)
    try:
        long = sc.get.rank_genes_groups_df(adata, group=None, key=_KEY)
    finally:
        adata.uns.pop(_KEY, None)
    long = long.rename(columns={"scores": "score", "pvals": "pval", "pvals_adj": "pval_adj"})
    return {
        str(group): table.set_index("names")[["score", "pval", "pval_adj"]].reindex(adata.var_names)
        for group, table in long.groupby("group", observed=True)
    }
