import numpy as np
import pytest

from stack_robustness.stats import paired_summary


def test_paired_summary_of_identical_differences_has_no_spread():
    out = paired_summary(np.full(7, 0.25), n_boot=200, seed=0)

    assert out["median"] == pytest.approx(0.25)
    assert out["ci_low"] == pytest.approx(0.25) and out["ci_high"] == pytest.approx(0.25)
    assert out["n"] == 7 and out["n_positive"] == 7


def test_paired_summary_interval_brackets_the_median_and_is_reproducible():
    diffs = np.random.default_rng(3).normal(0.1, 1.0, size=26)

    a = paired_summary(diffs, n_boot=500, seed=0)
    b = paired_summary(diffs, n_boot=500, seed=0)

    assert a == b
    assert a["ci_low"] <= a["median"] <= a["ci_high"]
    assert a["n_positive"] == int((diffs > 0).sum())


def test_paired_summary_ignores_missing_pairs():
    out = paired_summary(np.array([1.0, np.nan, 3.0]), n_boot=100, seed=0)

    assert out["n"] == 2 and out["median"] == pytest.approx(2.0)
