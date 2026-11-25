import anndata as ad
import numpy as np
import pandas as pd
import pytest

from stack_compositional.contexts import build_context

GENES = [f"g{i}" for i in range(40)]
CONDITIONS = ["control", "A", "B", "C", "A+B", "A+C", "B+C"]
N = 24  # cells per condition


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    cond = np.repeat(CONDITIONS, N)
    X = rng.gamma(2.0, 0.5, size=(len(cond), len(GENES)))
    for i, c in enumerate(CONDITIONS[1:]):
        X[cond == c, i] += 3.0  # each condition raises its own gene
    obs = pd.DataFrame({"condition": cond}, index=[f"cell{i}" for i in range(len(cond))])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))
    halves = pd.Series(np.tile(["first", "second"], len(cond) // 2), index=obs.index)
    return adata, halves


def _mean(adata, cond, mask=None):
    sel = (adata.obs["condition"] == cond).to_numpy()
    if mask is not None:
        sel = sel & mask
    return adata.X[sel].mean(axis=0)


# Two held-out pairs: the shared de.rank_genes needs at least two groups besides the reference
# (scanpy drops the "group" column when there is only one).
@pytest.fixture(scope="module")
def contexts(toy):
    adata, halves = toy
    kw = dict(test=["A+C", "B+C"], train=["A", "B", "C", "A+B"], halves=halves, n_hvg=10)
    return {t: build_context(adata, truth=t, **kw) for t in ("halves", "all")}


def test_halves_truth_is_the_first_half_and_all_truth_is_every_cell(toy, contexts):
    adata, halves = toy
    first = (halves == "first").to_numpy()
    np.testing.assert_allclose(contexts["halves"].truth.loc["A+C"].to_numpy(copy=True), _mean(adata, "A+C", first))
    np.testing.assert_allclose(contexts["all"].truth.loc["A+C"].to_numpy(copy=True), _mean(adata, "A+C"))


def test_only_halves_mode_has_duplicates(contexts):
    assert {"technical_duplicate", "interpolated_duplicate"} <= set(contexts["halves"].baselines)
    assert not {"technical_duplicate", "interpolated_duplicate"} & set(contexts["all"].baselines)
    for ctx in contexts.values():
        assert {"control_baseline", "mean_baseline", "additive"} <= set(ctx.baselines)


def test_inputs_do_not_depend_on_the_mode(contexts):
    h, a = contexts["halves"], contexts["all"]
    pd.testing.assert_series_equal(h.ctrl, a.ctrl)
    pd.testing.assert_frame_equal(h.baselines["additive"], a.baselines["additive"])
    assert h.panels == a.panels


def test_the_same_prediction_scores_differently_under_each_truth(contexts):
    pred = {"model": contexts["all"].baselines["additive"] + 0.1}
    s = {t: ctx.score(pred).query("model == 'model' and metric == 'mse'")["value"].iloc[0]
         for t, ctx in contexts.items()}
    assert s["halves"] != pytest.approx(s["all"])


def test_unknown_truth_mode_is_rejected(toy):
    adata, halves = toy
    with pytest.raises(ValueError):
        build_context(adata, ["A+C"], ["A"], halves, truth="thirds")
