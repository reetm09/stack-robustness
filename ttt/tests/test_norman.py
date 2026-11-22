import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from stack_ttt.norman import load_norman, single_pools

VOCAB = ["TP53", "C1ORF112", "A1BG", "NOTINDATA"]


@pytest.fixture
def raw_path(tmp_path):
    counts = np.array(
        [
            [1, 2, 3, 4],  # control
            [5, 6, 7, 8],  # A_B
            [9, 10, 11, 12],  # A
            [13, 14, 15, 16],  # B
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"perturbation": ["control", "A_B", "A", "B"], "ncounts": [100.0, 200.0, 300.0, 400.0]},
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["A1BG", "C1orf112", "XYZ", "TP53"])
    path = tmp_path / "norman.h5ad"
    ad.AnnData(X=sparse.csc_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return path


def test_load_norman_lines_genes_up_with_stack_vocabulary_ignoring_case(raw_path):
    cells = load_norman(raw_path, VOCAB, conditions={"A+B", "A"})

    assert list(cells.condition) == ["A+B", "A"]
    assert cells.genes == VOCAB
    # data columns A1BG, C1orf112, XYZ, TP53 -> vocabulary TP53, C1ORF112, A1BG, (missing)
    assert cells.counts.toarray().tolist() == [[8.0, 6.0, 5.0, 0.0], [12.0, 10.0, 9.0, 0.0]]
    assert cells.library.tolist() == [200.0, 300.0]
    assert cells.rows("A").tolist() == [1]


def test_load_norman_applies_gene_renames_before_matching_the_vocabulary(raw_path):
    cells = load_norman(raw_path, VOCAB, conditions={"A"}, aliases={"XYZ": "NOTINDATA"})

    # the data's XYZ column (11 for cell A) now fills the vocabulary's NOTINDATA position
    assert cells.counts.toarray().tolist() == [[12.0, 10.0, 9.0, 11.0]]


def test_single_pools_split_each_single_into_disjoint_adapt_and_infer_cells():
    condition = np.array(["A"] * 6 + ["B"] * 4 + ["A+B"] * 3)

    pools = single_pools(condition, singles=["A", "B"], seed=0)

    for single, n in [("A", 6), ("B", 4)]:
        adapt, infer = pools[(single, "adapt")], pools[(single, "infer")]
        assert len(adapt) + len(infer) == n and not set(adapt) & set(infer)
        assert set(adapt) | set(infer) == set(np.flatnonzero(condition == single))
    assert ("A+B", "adapt") not in pools
