"""In-context prediction with STACK, and turning generated cells into scored profiles."""

import contextlib
import io

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from stack_robustness.data import log_normalize


def generate(
    model,
    prompt,
    query,
    genes,
    genelist_path,
    num_steps: int,
    seed: int,
    batch_size: int = 8,
    prompt_ratio: float = 0.25,
    context_ratio: float = 0.4,
    context_ratio_min: float = 0.2,
):
    """STACK's own iterative in-context generation (arc-stack's get_incontext_generation,
    "mdm" mode, num_steps = T). `prompt` and `query` are cells x genes counts in the
    order of `genes`, STACK's vocabulary. Returns sampled counts, one row per query cell.
    """
    # arc-stack calls np.char.upper on var_names, which fails on pandas 3's string arrays.
    var = pd.DataFrame(index=pd.Index(list(genes), dtype=object))
    base = ad.AnnData(X=sparse.csr_matrix(prompt, dtype=np.float32), var=var)
    test = ad.AnnData(X=sparse.csr_matrix(query, dtype=np.float32), var=var.copy())
    with contextlib.redirect_stdout(io.StringIO()):  # it prints its schedules every step
        out = model.get_incontext_generation(
            base,
            test,
            str(genelist_path),
            num_steps=num_steps,
            prompt_ratio=prompt_ratio,
            context_ratio=context_ratio,
            context_ratio_min=context_ratio_min,
            mask_rate=1.0,
            mode="mdm",
            batch_size=batch_size,
            show_progress=False,
            num_workers=0,
            random_seed=seed,
        )
    return sparse.csr_matrix(out[0] if isinstance(out, tuple) else out)


def mean_log_profile(counts, library, genes, panel) -> pd.Series:
    """Mean of log1p(counts / library * 1e4) over cells, for the `panel` genes.

    `library` holds each cell's total over all genes, the convention used for the
    ground truth. For generated cells, pass the totals of the query cells they came from.
    """
    position = {g: i for i, g in enumerate(genes)}
    columns = [position[g] for g in panel]
    X = log_normalize(sparse.csr_matrix(counts, dtype=np.float64)[:, columns], library)
    return pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=list(panel))


def median_of_medians(profiles: dict) -> pd.Series:
    """Per gene, the median over draws within each transformation, then the median
    over transformations (less sensitive to one poorly behaved transformation)."""
    within = [pd.concat(draws, axis=1).median(axis=1) for draws in profiles.values()]
    return pd.concat(within, axis=1).median(axis=1)
