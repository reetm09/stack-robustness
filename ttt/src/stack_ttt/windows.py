"""Windows of cells in STACK's fine-tuning layout, for test-time adaptation."""

import numpy as np


def pick(pool, n: int, rng) -> np.ndarray:
    """n rows of `pool`: without replacement when it has n or more, else every row once
    plus random repeats."""
    pool = np.asarray(pool)
    if len(pool) >= n:
        return rng.choice(pool, size=n, replace=False)
    return np.concatenate([rng.permutation(pool), rng.choice(pool, size=n - len(pool), replace=True)])


def task_window(counts, prompt_rows, query_rows, target_rows, n_cells: int, n_kept: int, rng):
    """(observed, truth) arrays of n_cells x genes counts.

    The first n_kept rows are prompt cells, identical in both. The rest are query cells
    in `observed` and target cells in `truth`. With no query rows, every position is a
    prompt cell (n_kept must equal n_cells).
    """
    prompt = counts[pick(prompt_rows, n_kept, rng)].toarray().astype(np.float32)
    if query_rows is None:
        if n_kept != n_cells:
            raise ValueError("a prompt-only window needs n_kept == n_cells")
        return prompt, prompt.copy()
    n_query = n_cells - n_kept
    query = counts[pick(query_rows, n_query, rng)].toarray().astype(np.float32)
    target = counts[pick(target_rows, n_query, rng)].toarray().astype(np.float32)
    return np.vstack([prompt, query]), np.vstack([prompt, target])
