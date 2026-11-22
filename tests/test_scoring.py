import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score

from stack_robustness.scoring import METRICS, relative_to_negative, score_profiles

GENES = ["g0", "g1", "g2", "g3", "g4"]
CTRL = np.array([1.0, 3.0, 2.0, 3.0, 1.0])
PERT_REF = np.array([1.0, 4.0, 2.0, 5.0, 1.0])  # differs from control on g1 and g3
PRED_DCTRL = np.array([2.0, 2.0, 0.0, -1.0, 1.0])  # P1's predicted change from control
TRUTH_DCTRL = np.array([3.0, 1.0, 0.0, -2.0, 0.0])  # P1's true change from control


def _de(rows):
    """DE table from {gene: (score, pval_adj)}; rows are out of panel order and include
    gX, a gene outside the scored panel."""
    scores, pvals = zip(*rows.values())
    return pd.DataFrame({"score": scores, "pval": pvals, "pval_adj": pvals}, index=list(rows))


# Against control, g0, g1 and g3 are DE; g0 and g3 have the smallest p-values.
DE_CTRL = _de({"g3": (-3.0, 0.01), "gX": (50.0, 1e-9), "g1": (1.0, 0.02), "g4": (0.2, 0.5), "g0": (5.0, 0.01), "g2": (0.0, 0.5)})
# Against the rest, the top three panel genes are g1, g2 and g3; gX would lead if it counted.
DE_REST = _de({"g2": (-3.0, 0.01), "g4": (0.5, 0.5), "gX": (0.0, 1e-12), "g0": (0.1, 0.5), "g3": (2.0, 0.03), "g1": (4.0, 0.01)})


def _corr(x, y):
    return np.corrcoef(x, y)[0, 1]


@pytest.fixture
def scored():
    far = np.full(5, 9.0)
    truth = pd.DataFrame([CTRL + TRUTH_DCTRL, far], index=["P1", "P2"], columns=GENES)
    pred = pd.DataFrame([CTRL + PRED_DCTRL, far], index=["P1", "P2"], columns=GENES)
    return score_profiles(
        pred,
        truth,
        pd.Series(CTRL, index=GENES),
        pd.Series(PERT_REF, index=GENES),
        de_vs_ctrl={"P1": DE_CTRL, "P2": DE_CTRL},
        de_vs_rest={"P1": DE_REST, "P2": DE_REST},
        top_n=3,
        overlap_k=2,
    )


def test_delta_metrics_use_control_and_mean_perturbed_references(scored):
    p1 = scored.loc["P1"]
    assert p1["pearson_deltactrl"] == pytest.approx(_corr(PRED_DCTRL, TRUTH_DCTRL))
    assert p1["pearson_deltapert"] == pytest.approx(_corr([2, 1, 0, -3, 1], [3, 0, 0, -4, 0]))
    assert p1["mse"] == pytest.approx(0.8)
    assert p1["mae"] == pytest.approx(0.8)


def test_weights_and_top_genes_come_from_de_against_the_rest_within_the_panel(scored):
    p1 = scored.loc["P1"]
    # Panel |scores| 0.1, 4, 3, 2, 0.5 min-max scaled then squared; P1 misses every gene but g2 by 1.
    assert p1["wmse"] == pytest.approx((3.9**2 + 1.9**2 + 0.4**2) / (3.9**2 + 2.9**2 + 1.9**2 + 0.4**2))
    # Top three by p-value against the rest: g1, g2, g3.
    assert p1["pearson_deltactrl_top"] == pytest.approx(_corr([2, 0, -1], [1, 0, -2]))
    assert p1["pearson_deltapert_top"] == pytest.approx(_corr([1, 0, -3], [0, 0, -4]))


def test_cell_eval_metrics_use_de_against_control(scored):
    p1 = scored.loc["P1"]
    is_de = np.array([True, True, False, True, False])
    assert p1["de_pr_auc"] == pytest.approx(average_precision_score(is_de, np.abs(PRED_DCTRL)))
    assert p1["spearman_deltactrl_de"] == pytest.approx(stats.spearmanr([2, 2, -1], [3, 1, -2]).statistic)
    assert p1["direction_agreement"] == pytest.approx(1.0)
    # True top two (g0, g3) against the two largest predicted changes (g0, g1).
    assert p1["top_overlap"] == pytest.approx(0.5)


def test_nir_compares_each_prediction_with_every_perturbations_truth(scored):
    assert scored["nir"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("metric", sorted(METRICS))
def test_each_metric_is_judged_against_its_own_negative_control(metric):
    info = METRICS[metric]
    away = 1.0 if info.higher_is_better else -1.0  # direction away from a perfect score
    p = ["P1"]
    scores = {
        "control": pd.DataFrame({metric: [info.perfect - 0.8 * away]}, index=p),
        "mean": pd.DataFrame({metric: [info.perfect - 0.4 * away]}, index=p),
        "model": pd.DataFrame({metric: [info.perfect - 0.2 * away]}, index=p),
    }

    out = relative_to_negative(scores, mean_baseline="mean", control_baseline="control")

    # Miller et al. judge the Delta-pert metrics against the control baseline, because the
    # mean baseline is their reference and scores NaN; every other metric against the mean.
    expected = 0.75 if "deltapert" in metric else 0.5
    assert out["model"].loc["P1", metric] == pytest.approx(expected, abs=1e-5)
