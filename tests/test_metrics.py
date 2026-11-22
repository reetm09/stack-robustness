import math

import numpy as np
import pytest

from stack_robustness.metrics import (
    de_pr_auc,
    deg_weights,
    direction_agreement,
    mae,
    mse,
    nir,
    pearson_delta,
    spearman_on,
    top_k_overlap,
    wmse,
)


def test_pearson_delta_correlates_changes_from_the_reference_not_raw_profiles():
    ref = np.array([0.0, 0.0, 6.0])
    pred = ref + np.array([1.0, 0.0, 0.0])
    truth = ref + np.array([0.0, 1.0, 0.0])

    assert pearson_delta(pred, truth, ref) == pytest.approx(-0.5)


def test_pearson_delta_is_nan_when_the_prediction_does_not_move_from_the_reference():
    ref = np.array([1.0, 2.0, 3.0])
    truth = np.array([2.0, 2.0, 4.0])

    assert math.isnan(pearson_delta(ref.copy(), truth, ref))


def test_mse_and_mae_average_over_genes():
    pred = np.array([1.0, 3.0])
    truth = np.array([0.0, 0.0])

    assert mse(pred, truth) == 5.0
    assert mae(pred, truth) == 2.0


def test_wmse_normalizes_the_weights_to_sum_to_one():
    pred = np.array([1.0, 2.0])
    truth = np.array([0.0, 0.0])
    weights = np.array([1.0, 3.0])

    assert wmse(pred, truth, weights) == pytest.approx(0.25 * 1.0 + 0.75 * 4.0)


def test_deg_weights_are_squared_min_max_scaled_absolute_scores():
    scores = np.array([-4.0, 2.0, 0.0, np.nan])

    assert deg_weights(scores).tolist() == [1.0, 0.25, 0.0, 0.0]


def test_spearman_on_uses_only_the_masked_genes():
    pred = np.array([1.0, 2.0, 3.0, 100.0])
    truth = np.array([1.0, 3.0, 2.0, 0.0])
    mask = np.array([True, True, True, False])

    assert spearman_on(pred, truth, mask) == pytest.approx(0.5)


def test_de_pr_auc_ranks_genes_by_absolute_predicted_change():
    pred_delta = np.array([0.9, 0.1, 0.8, -0.95])
    is_de = np.array([True, False, False, True])

    assert de_pr_auc(pred_delta, is_de) == pytest.approx(1.0)


def test_de_pr_auc_is_nan_without_any_true_de_gene():
    assert math.isnan(de_pr_auc(np.array([0.5, 0.1]), np.array([False, False])))


def test_direction_agreement_counts_sign_matches_on_masked_genes():
    pred_delta = np.array([1.0, -1.0, 2.0, 0.5])
    truth_delta = np.array([2.0, 1.0, -1.0, 3.0])
    mask = np.array([True, True, False, True])

    assert direction_agreement(pred_delta, truth_delta, mask) == pytest.approx(2 / 3)


def test_top_k_overlap_compares_largest_predicted_changes_with_true_top_genes():
    pred_delta = np.array([5.0, -4.0, 0.1, 3.0, 0.0])
    true_top = np.array([False, True, True, False, False])

    assert top_k_overlap(pred_delta, true_top) == pytest.approx(0.5)


def test_nir_is_the_fraction_of_other_truths_farther_than_the_matching_truth():
    truth = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    pred = np.array([[1.0, 0.0], [0.0, 9.0], [0.0, 6.0]])

    assert nir(pred, truth).tolist() == [1.0, 0.0, 1.0]
