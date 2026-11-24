"""Windows of cells in STACK's fine-tuning layout, for test-time adaptation."""

import numpy as np


def pick(pool, n: int, rng) -> np.ndarray:
    """n rows of `pool`: without replacement when it has n or more, else every row once
    plus random repeats."""
    pool = np.asarray(pool)
    if len(pool) >= n:
        return rng.choice(pool, size=n, replace=False)
    return np.concatenate([rng.permutation(pool), rng.choice(pool, size=n - len(pool), replace=True)])


def task_window(counts, prompt_rows, query_rows, target_rows, n_cells: int, n_kept: int, rng, reveal: str = "target"):
    """(observed, truth, revealed) arrays of n_cells x genes counts.

    The first n_kept rows are prompt cells, identical in all three. The rest are query cells
    in `observed` and target cells in `truth`. The loss reveals a random share of the
    positions after the prompt (objective.ttt_loss), showing:
    - "target": their target cells, as in STACK's post-training, where those share the
      prompt's condition; `revealed` is None;
    - "prompt": more prompt cells, drawn with the prompt cells as one sample, which is what
      STACK's inference puts there; `revealed` holds them.
    With no query rows, every position is a prompt cell (n_kept must equal n_cells).
    """
    if reveal not in ("target", "prompt"):
        raise ValueError(f"unknown reveal {reveal!r}")
    if query_rows is None:
        if n_kept != n_cells:
            raise ValueError("a prompt-only window needs n_kept == n_cells")
        prompt = counts[pick(prompt_rows, n_kept, rng)].toarray().astype(np.float32)
        return prompt, prompt.copy(), None
    n_query = n_cells - n_kept
    if reveal == "prompt":
        drawn = counts[pick(prompt_rows, n_cells, rng)].toarray().astype(np.float32)
        prompt, revealed = drawn[:n_kept], drawn
    else:
        prompt, revealed = counts[pick(prompt_rows, n_kept, rng)].toarray().astype(np.float32), None
    query = counts[pick(query_rows, n_query, rng)].toarray().astype(np.float32)
    target = counts[pick(target_rows, n_query, rng)].toarray().astype(np.float32)
    return np.vstack([prompt, query]), np.vstack([prompt, target]), revealed
