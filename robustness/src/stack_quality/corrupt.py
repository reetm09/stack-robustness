"""Count-level corruptions (proposal, Robustness: "Technical corruption").

Every function takes raw counts (cells x genes, every measured gene rather than only STACK's
vocabulary) and returns new CSR float32 counts, leaving the input untouched. Library sizes
downstream are the corrupted matrix's row sums, so normalisation sees the corrupted depth.
"""

import numpy as np
from scipy import sparse


def thin(X, rho: float, rng: np.random.Generator) -> sparse.csr_matrix:
    """Binomial thinning: each count kept with probability 1 - rho (x~ ~ Binomial(x, 1 - rho))."""
    X = sparse.csr_matrix(X)
    if rho <= 0:
        return X.astype(np.float32, copy=True)
    kept = rng.binomial(np.rint(X.data).astype(np.int64), 1.0 - rho).astype(np.float32)
    out = sparse.csr_matrix((kept, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


def background_profile(X) -> np.ndarray:
    """Empirical ambient profile b: each gene's share of all counts in X."""
    totals = np.asarray(sparse.csr_matrix(X).sum(axis=0), dtype=np.float64).ravel()
    return totals / totals.sum()


def ambient(X, alpha: float, background: np.ndarray, rng: np.random.Generator, chunk: int = 256) -> sparse.csr_matrix:
    """Ambient contamination at fixed library size:
    x~_c ~ Multinomial(N_c, (1 - alpha) x_c / N_c + alpha b), with N_c the cell's total count."""
    X = sparse.csr_matrix(X)
    if alpha <= 0:
        return X.astype(np.float32, copy=True)
    b = np.asarray(background, dtype=np.float64)
    library = np.rint(np.asarray(X.sum(axis=1)).ravel()).astype(np.int64)
    blocks = []
    for start in range(0, X.shape[0], chunk):
        rows = X[start : start + chunk].toarray().astype(np.float64)
        n = library[start : start + chunk]
        p = (1.0 - alpha) * rows / np.maximum(n, 1)[:, None] + alpha * b[None, :]
        p /= p.sum(axis=1, keepdims=True)
        p *= 1.0 - 1e-12  # numpy rejects sum(p[:-1]) > 1 by rounding error
        sampled = rng.multinomial(n, p)  # rows with n = 0 stay all-zero
        blocks.append(sparse.csr_matrix(sampled.astype(np.float32)))
    return sparse.vstack(blocks, format="csr")


def corrupt(X, *, thin_rate: float = 0.0, ambient_rate: float = 0.0, background=None, rng) -> sparse.csr_matrix:
    """Thinning, then ambient mixing (either may be zero)."""
    out = thin(X, thin_rate, rng)
    if ambient_rate > 0:
        out = ambient(out, ambient_rate, background, rng)
    return out
