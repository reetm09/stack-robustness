import numpy as np
import pandas as pd

import stack_quality.tabular as tabular
from stack_quality.score import cluster_bootstrap, summarise_arm


def test_lodo_never_trains_on_the_held_out_drug(monkeypatch):
    seen = []

    def fake(name, X, Y, X_test, **kw):
        seen.append(X[:, -2].copy())  # second-to-last feature column encodes the drug below
        return np.zeros((len(X_test), Y.shape[1]))

    monkeypatch.setattr(tabular, "_fit_predict", fake)
    comps = np.eye(3)
    drugs = ["a", "a", "b", "b", "c", "c"]
    train = pd.DataFrame({"drug": drugs, "kind": ["B", "My"] * 3,
                          "x": [np.array([0, 0, i]) for i in (0, 0, 1, 1, 2, 2)], "y": [np.ones(3)] * 6})
    test = pd.DataFrame({"drug": ["a", "c"], "kind": ["B", "B"], "x": [np.zeros(3)] * 2})
    tabular.lodo_predict("ridge", train, test, comps, log=lambda *_: None)
    assert sorted(set(seen[0])) == [1, 2]  # drug a held out
    assert sorted(set(seen[1])) == [0, 1]  # drug c held out


def test_ridge_runs_and_ceiling_is_a_projection():
    rng = np.random.default_rng(0)
    comps = np.linalg.qr(rng.normal(size=(10, 4)))[0].T  # 4 orthonormal components over 10 genes
    train = pd.DataFrame({"drug": list("aabbccdd"), "kind": ["B", "My"] * 4,
                          "x": list(rng.normal(size=(8, 10))), "y": list(rng.normal(size=(8, 10)))})
    test = pd.DataFrame({"drug": ["a"], "kind": ["B"], "x": [rng.normal(size=10)]})
    pred = tabular.lodo_predict("ridge", train, test, comps, log=lambda *_: None)
    assert pred.shape == (1, 10)
    v = rng.normal(size=(1, 10))
    once = tabular.pca_ceiling(v, comps)
    np.testing.assert_allclose(tabular.pca_ceiling(once, comps), once, atol=1e-12)


def test_cluster_bootstrap_resamples_drugs():
    t = pd.DataFrame({"perturbation": ["x|B|p", "x|My|p", "y|B|p"], "v": [1.0, 1.0, 1.0]})
    assert cluster_bootstrap(t, "v", n_boot=50) == (1.0, 1.0, 1.0)


def test_degradation_is_positive_when_a_higher_is_better_score_drops():
    rows = []
    for target in ("d1|B|p", "d2|B|p"):
        for sev, val in ((0.0, 0.8), (0.5, 0.5)):
            rows.append({"arm": "thinning", "target": "both", "model": "m", "perturbation": target,
                         "severity": sev, "value": val})
    out = summarise_arm(pd.DataFrame(rows), "pearson_deltactrl", n_boot=20)
    assert out.loc[0, "degradation@0.5"].startswith("0.300")
    assert abs(out.loc[0, "worst"] - 0.5) < 1e-12
