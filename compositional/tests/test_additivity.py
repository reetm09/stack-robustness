import numpy as np
import pandas as pd
import pytest

from stack_compositional.additivity import calibrated, specific_epistasis_fit


def test_calibrated_maps_duplicate_to_0_and_negative_to_1():
    assert calibrated(0.2, 0.2, 1.0) == 0.0
    assert calibrated(1.0, 0.2, 1.0) == pytest.approx(1.0)
    assert calibrated(0.6, 0.2, 1.0) == pytest.approx(0.5)
    assert calibrated(0.1, 0.2, 1.0) == 0.0          # better than the duplicate: clipped
    assert calibrated(2.0, 0.2, 1.0) == pytest.approx(2.25)  # worse than uninformative: kept
    assert np.isnan(calibrated(0.5, 1.0, 0.8))       # duplicate not better than negative: dropped


def test_calibrated_vectorised():
    out = calibrated([0.2, 0.6], [0.2, 0.2], [1.0, 1.0])
    np.testing.assert_allclose(out, [0.0, 0.5])


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(60)]
    singles = ["A", "B", "C", "D"]
    effects = pd.DataFrame(rng.normal(size=(4, 60)), index=singles, columns=genes)
    ctrl = pd.Series(rng.normal(size=60) + 2.0, index=genes)
    eps = pd.Series(rng.normal(scale=0.3, size=60), index=genes)
    true_c = {"A+B": (1.0, 1.0), "A+C": (0.5, 0.4), "B+D": (1.5, 0.0), "C+D": (0.0, 1.0), "A+D": (0.8, 1.2)}
    doubles = pd.DataFrame({p: ctrl + c1 * effects.loc[p[0]] + c2 * effects.loc[p[2]] + eps
                            for p, (c1, c2) in true_c.items()}).T
    return ctrl, effects, doubles, true_c, eps


def test_specific_epistasis_recovers_pair_weights_and_shared_offset(synthetic):
    ctrl, effects, doubles, true_c, eps = synthetic
    coefs, eps_hat, fitted = specific_epistasis_fit(ctrl, effects, doubles)
    for pair, (c1, c2) in true_c.items():
        assert coefs.loc[pair, "c1"] == pytest.approx(c1, abs=1e-5)
        assert coefs.loc[pair, "c2"] == pytest.approx(c2, abs=1e-5)
    np.testing.assert_allclose(eps_hat.to_numpy(), eps.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(fitted.to_numpy(), doubles.to_numpy(), atol=1e-5)


def test_specific_epistasis_never_fits_worse_than_plain_addition(synthetic):
    ctrl, effects, doubles, _, _ = synthetic
    noisy = doubles + np.random.default_rng(1).normal(scale=0.5, size=doubles.shape)
    _, _, fitted = specific_epistasis_fit(ctrl, effects, noisy)
    additive = pd.DataFrame({p: ctrl + effects.loc[p[0]] + effects.loc[p[2]] for p in noisy.index}).T
    assert ((fitted - noisy) ** 2).to_numpy().sum() <= ((additive - noisy) ** 2).to_numpy().sum()


def test_additivity_groups_top_quarter_with_one_cut():
    from stack_compositional.additivity import additivity_groups
    plain = pd.Series({f"p{i}": v for i, v in enumerate([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.5, 0.9])})
    stretched = pd.Series({f"p{i}": v for i, v in enumerate([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.01, 0.6])})
    labels, cut = additivity_groups(plain, stretched)
    assert cut == pytest.approx(plain.quantile(0.75))
    assert labels["p7"] == "interaction"        # far even after stretching
    assert labels["p6"] == "buffering/synergy"  # stretching fixes it
    assert labels["p1"] == "additive"           # plain sum fine: stretched score ignored
    assert (labels == "additive").sum() == 6


def test_additivity_groups_keep_missing_scores_missing():
    from stack_compositional.additivity import additivity_groups
    labels, _ = additivity_groups(pd.Series({"a": 0.1, "b": np.nan, "c": 0.5, "d": 0.0}),
                                  pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}))
    assert pd.isna(labels["b"])
