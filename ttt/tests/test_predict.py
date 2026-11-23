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


def _two_singles(tmp_path):
    genelist = tmp_path / "genes.pkl"
    genelist.write_bytes(pickle.dumps(GENES))
    rng = np.random.default_rng(0)
    condition = np.array(["A"] * 24 + ["B"] * 24 + ["control"] * 24)
    counts = sparse.csr_matrix(rng.poisson(3.0, size=(72, len(GENES))).astype(np.float32))
    cells = Cells(counts=counts, library=np.full(72, 500.0), condition=condition, genes=GENES)
    return cells, single_pools(condition, ["A", "B"], seed=0), genelist


def test_predict_double_reports_raw_and_unchanged_query_profiles(tiny_model, tmp_path):
    cells, pools, genelist = _two_singles(tmp_path)

    out = predict_double(tiny_model, cells, pools, "A+B", genelist, panel=GENES[:5], num_steps=1, draws=2)

    assert set(out) == {"raw", "raw_one", "query", "query_one"}
    assert all(list(v.index) == GENES[:5] for v in out.values())
    b_rows = pools[("B", "infer")]
    expected = mean_log_profile(cells.counts[b_rows], cells.library[b_rows], GENES, GENES[:5])
    pd.testing.assert_series_equal(out["query_one"], expected, check_names=False)


def test_differential_prediction_is_the_query_itself_when_the_prompt_is_the_reference(tiny_model, tmp_path):
    cells, pools, genelist = _two_singles(tmp_path)
    reference = pools[("A", "infer")]  # prompting with A is then identical to the reference prompt

    out = predict_double(tiny_model, cells, pools, "A+B", genelist, panel=GENES[:5], num_steps=1, draws=2,
                         reference=reference)

    assert {"diff", "diff_one"} <= set(out)
    pd.testing.assert_series_equal(out["diff_one"], out["query_one"], check_names=False)


def test_stack_generation_tolerates_its_unmask_rate_rounding_just_above_one():
    # arc-stack 0.1.3's last generation step asks np.quantile for q = (m/n - 0) * n/m, which is
    # 1.0000000000000002 for m=7, n=25 (and ~3% of pairs); stack_ttt clamps it to 1.
    from stack.models.core import inference

    values = np.arange(7.0)
    q = (7 / 25 - 0.0) * 25 / 7

    assert q > 1.0
    assert inference.np.quantile(values, q) == 6.0
    assert inference.np.arange(3).tolist() == [0, 1, 2]  # everything else is plain numpy


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


@pytest.mark.parametrize("n_query, num_steps", [(2, 1), (3, 2)])
def test_generation_handles_query_sets_smaller_than_half_a_window(tiny_model, tmp_path, n_query, num_steps):
    # arc-stack 0.1.3 pads a short query set with at most one copy of itself, so with fewer
    # query cells than half a window's query slots (here 5 at T=1, 8 at the first of 2 steps)
    # the window comes up short and its own bookkeeping raises IndexError.
    genelist = tmp_path / "genes.pkl"
    genelist.write_bytes(pickle.dumps(GENES))
    rng = np.random.default_rng(0)
    prompt = sparse.csr_matrix(rng.poisson(4.0, size=(30, len(GENES))).astype(np.float32))
    query = sparse.csr_matrix(rng.poisson(2.0, size=(n_query, len(GENES))).astype(np.float32))

    out = generate(tiny_model, prompt, query, GENES, genelist, num_steps=num_steps, seed=0)

    assert out.shape == (n_query, len(GENES))
