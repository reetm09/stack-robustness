import numpy as np
from scipy import sparse

from stack_quality.corrupt import ambient, background_profile, thin


def counts(seed=0, shape=(200, 50), lam=5.0):
    return sparse.csr_matrix(np.random.default_rng(seed).poisson(lam, size=shape).astype(np.float32))


def test_zero_severity_is_identity():
    X = counts()
    b = background_profile(X)
    assert (thin(X, 0.0, np.random.default_rng(1)) != X).nnz == 0
    assert (ambient(X, 0.0, b, np.random.default_rng(1)) != X).nnz == 0


def test_thinning_keeps_the_expected_share_and_never_adds():
    X = counts()
    Y = thin(X, 0.25, np.random.default_rng(1))
    assert (Y.toarray() <= X.toarray()).all()
    assert abs(Y.sum() / X.sum() - 0.75) < 0.01
    assert X.sum() == counts().sum()  # input untouched


def test_ambient_keeps_library_sizes_and_moves_toward_background():
    X = counts(lam=3.0)
    b = np.zeros(X.shape[1])
    b[0] = 1.0  # all ambient counts land on gene 0
    Y = ambient(X, 0.2, b, np.random.default_rng(2), chunk=64)
    np.testing.assert_array_equal(np.asarray(Y.sum(axis=1)).ravel(), np.asarray(X.sum(axis=1)).ravel())
    share0 = Y[:, 0].sum() / Y.sum()
    expected = 0.8 * X[:, 0].sum() / X.sum() + 0.2
    assert abs(share0 - expected) < 0.01


def test_ambient_handles_empty_cells():
    X = counts().tolil()
    X[3, :] = 0
    Y = ambient(X.tocsr(), 0.1, background_profile(X), np.random.default_rng(0))
    assert Y[3].sum() == 0
