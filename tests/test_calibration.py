import math

import pytest

from stack_robustness.calibration import drf


def test_drf_for_a_higher_is_better_metric_is_the_share_of_the_gap_to_perfect():
    assert drf(pos=0.8, neg=0.2, perfect=1.0, higher_is_better=True) == pytest.approx(0.75, abs=1e-5)


def test_drf_for_an_error_metric_is_the_share_of_the_negative_controls_error_removed():
    assert drf(pos=1.0, neg=4.0, perfect=0.0, higher_is_better=False) == pytest.approx(0.75, abs=1e-5)


def test_drf_is_negative_when_the_positive_control_loses_to_the_negative_control():
    assert drf(pos=0.1, neg=0.2, perfect=1.0, higher_is_better=True) == pytest.approx(-0.125, abs=1e-5)


def test_drf_is_clipped_to_minus_one():
    assert drf(pos=10.0, neg=1.0, perfect=0.0, higher_is_better=False) == -1.0


def test_drf_is_nan_when_a_control_score_is_missing():
    assert math.isnan(drf(pos=float("nan"), neg=0.2, perfect=1.0, higher_is_better=True))
    assert math.isnan(drf(pos=0.5, neg=float("nan"), perfect=1.0, higher_is_better=True))
