import anndata as ad
import numpy as np
import pandas as pd

from stack_robustness.de import rank_genes

GENES = ["g0", "g1", "g2", "g3"]


def _toy_adata():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1.0, scale=0.1, size=(60, 4))
    X[20:40, 2] += 2.0  # group A raises g2
    X[40:60, 0] -= 0.5  # group B lowers g0
    obs = pd.DataFrame({"cond": ["ctrl"] * 20 + ["A"] * 20 + ["B"] * 20}, index=[f"c{i}" for i in range(60)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))


def test_rank_genes_returns_tables_aligned_to_the_gene_order():
    tables = rank_genes(_toy_adata(), "cond", reference="rest")

    a = tables["A"]
    assert a.index.tolist() == GENES
    assert a["pval_adj"].idxmin() == "g2"
    assert a.loc["g2", "score"] > 0
    assert tables["B"].loc["g0", "score"] < 0


def test_rank_genes_against_a_reference_group_leaves_that_group_out():
    tables = rank_genes(_toy_adata(), "cond", reference="ctrl", method="wilcoxon")

    assert sorted(tables) == ["A", "B"]
    assert tables["A"]["pval_adj"].idxmin() == "g2"


def test_rank_genes_does_not_leave_results_on_the_input():
    adata = _toy_adata()

    rank_genes(adata, "cond", reference="rest")

    assert "rank_genes_groups" not in adata.uns
