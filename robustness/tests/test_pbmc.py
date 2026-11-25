import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from stack.cli.generation import _align_genes_to_target_list

from stack_quality.pbmc import eligible_targets, vocab_projection


def test_projection_matches_arc_alignment():
    genes = ["Abc", "DEF", "ghi", "JKL"]
    vocab = ["DEF", "abc", "NOPE", "JKL"]  # case differs, one missing, one data gene unused
    X = sparse.csr_matrix(np.arange(1, 13, dtype=np.float32).reshape(3, 4))
    ours = (X @ vocab_projection(genes, vocab)).toarray()
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=pd.Index(genes, dtype=object)))
    theirs = _align_genes_to_target_list(adata, vocab, None).X
    np.testing.assert_array_equal(ours, np.asarray(sparse.csr_matrix(theirs).toarray()))


def test_targets_need_enough_cells_and_plate_controls(cells):
    found = {t.id for t in eligible_targets(cells, min_cells=30)}
    assert found == {"A|B|plate_1", "A|My|plate_1", "A|B|plate_2", "A|My|plate_2", "B|B|plate_1", "B|My|plate_1"}
    assert eligible_targets(cells, min_cells=50) == []
