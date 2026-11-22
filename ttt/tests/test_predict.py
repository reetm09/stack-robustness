import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from stack_ttt.norman import Cells, single_pools
from stack_ttt.predict import generate, mean_log_profile, median_of_medians, predict_double

from conftest import TINY_CONFIG

GENES = [f"G{i}" for i in range(TINY_CONFIG["n_genes"])]


def test_median_of_medians_takes_medians_within_then_across_transformations():
    def s(v):
        return pd.Series([v], index=["g0"])

    profiles = {"identity": [s(1.0), s(2.0), s(9.0)], "swap": [s(4.0), s(6.0)]}

    # within: median(1, 2, 9) = 2 and median(4, 6) = 5; across: median(2, 5) = 3.5
    assert median_of_medians(profiles)["g0"] == pytest.approx(3.5)


def test_mean_log_profile_normalizes_by_the_given_library_and_keeps_panel_genes():
    counts = sparse.csr_matrix(np.array([[1.0, 3.0, 5.0], [0.0, 2.0, 7.0]]))
    library = np.array([10.0, 20.0])  # totals over all genes, not only these three

    out = mean_log_profile(counts, library, genes=["a", "b", "c"], panel=["c", "a"])

    assert list(out.index) == ["c", "a"]
    assert out["a"] == pytest.approx((math.log1p(1000.0) + 0.0) / 2)
    assert out["c"] == pytest.approx((math.log1p(5000.0) + math.log1p(3500.0)) / 2)


def test_predict_double_aggregates_both_prompt_orders(tiny_model, tmp_path):
    genelist = tmp_path / "genes.pkl"
    genelist.write_bytes(pickle.dumps(GENES))
    rng = np.random.default_rng(0)
    condition = np.array(["A"] * 24 + ["B"] * 24)
    counts = sparse.csr_matrix(rng.poisson(3.0, size=(48, len(GENES))).astype(np.float32))
    cells = Cells(counts=counts, library=np.full(48, 500.0), condition=condition, genes=GENES)
    pools = single_pools(condition, ["A", "B"], seed=0)

    out = predict_double(tiny_model, cells, pools, "A+B", genelist, panel=GENES[:5], num_steps=1, draws=2)

    assert set(out) == {"one", "both"}
    assert list(out["one"].index) == GENES[:5] and list(out["both"].index) == GENES[:5]
    assert np.isfinite(out["both"].to_numpy()).all()


def test_generate_returns_one_generated_cell_per_query_cell(tiny_model, tmp_path):
    genelist = tmp_path / "genes.pkl"
    genelist.write_bytes(pickle.dumps(GENES))
    rng = np.random.default_rng(0)
    prompt = sparse.csr_matrix(rng.poisson(4.0, size=(30, len(GENES))).astype(np.float32))
    query = sparse.csr_matrix(rng.poisson(2.0, size=(12, len(GENES))).astype(np.float32))

    out = generate(tiny_model, prompt, query, GENES, genelist, num_steps=2, seed=0)

    assert out.shape == (12, len(GENES))
    dense = out.toarray()
    assert (dense >= 0).all() and np.allclose(dense, np.round(dense))
