import numpy as np
import pytest
from scipy import sparse

from stack_quality.pbmc import DMSO, Cells, vocab_projection


@pytest.fixture
def cells():
    """Two plates; DMSO and drug A on both, drug B on plate_1 only; T, B and myeloid cells."""
    rng = np.random.default_rng(0)
    drug, kind, plate = [], [], []
    layout = {(DMSO, "plate_1"): 60, (DMSO, "plate_2"): 60, ("A", "plate_1"): 40, ("A", "plate_2"): 40,
              ("B", "plate_1"): 40}
    for (d, p), n in layout.items():
        for k in ("T", "B", "My"):
            drug += [d] * n
            kind += [k] * n
            plate += [p] * n
    genes = [f"G{i}" for i in range(30)]
    vocab = [g.lower() for g in genes[:20]] + ["MISSING1", "MISSING2"]
    counts = sparse.csr_matrix(rng.poisson(2.0, size=(len(drug), len(genes))).astype(np.float32))
    return Cells(counts, genes, np.array(drug), np.array(kind), np.array(plate), vocab, vocab_projection(genes, vocab))
