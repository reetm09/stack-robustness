"""In-context prediction with STACK, and turning generated cells into scored profiles."""

import contextlib
import io

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from stack.models.core import inference as _stack_inference

from stack_robustness.data import log_normalize


class _NumpyWithClampedQuantile:
    """numpy as arc-stack's inference module sees it, except that quantile clamps q into [0, 1].

    arc-stack 0.1.3's last generation step asks for q = (m/n - 0) * n/m, which floating point
    rounds to 1.0000000000000002 for about 3% of (masked, total) cell counts, and np.quantile
    then raises. The intended q is exactly 1: generate every cell still masked.
    """

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def quantile(a, q, *args, **kwargs):
        return np.quantile(a, np.clip(q, 0.0, 1.0), *args, **kwargs)


_stack_inference.np = _NumpyWithClampedQuantile()


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
    # arc-stack 0.1.3 fills a short window with at most one more copy of the query cells
    # (get_incontext_prediction: pad = test_indices[:need]), so a query set smaller than half
    # a window's query slots leaves the window short and raises IndexError. Repeat such a set
    # until it fills a window; the first copy's generated cells are returned.
    n_query = query.shape[0]
    ratios = [context_ratio] if num_steps == 1 else [context_ratio_min, context_ratio]
    slots = max(max(1, int(model.n_cells * (1 - prompt_ratio - r))) for r in ratios)
    if 2 * n_query < slots:
        query = sparse.vstack([sparse.csr_matrix(query)] * -(-slots // n_query))
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
    return sparse.csr_matrix(out[0] if isinstance(out, tuple) else out)[:n_query]


def mean_log_profile(counts, library, genes, panel) -> pd.Series:
    """Mean of log1p(counts / library * 1e4) over cells, for the `panel` genes.

    `library` holds each cell's total over all genes, the convention used for the
    ground truth. For generated cells, pass the totals of the query cells they came from.
    """
    position = {g: i for i, g in enumerate(genes)}
    columns = [position[g] for g in panel]
    X = log_normalize(sparse.csr_matrix(counts, dtype=np.float64)[:, columns], library)
    return pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=list(panel))


def predict_double(model, cells, pools, double, genelist_path, panel, num_steps, draws, batch_size=8,
                   reference=None) -> dict:
    """Predicted profiles for a double A+B, from the "infer" cells of A and B.

    For each prompt order (A prompts B's cells, and B prompts A's):
      raw    the generated cells' profile
      query  the query cells' own observed profile (no model), the floor STACK must beat
      diff   (with `reference`, rows of control cells) the observed query profile plus
             generated-with-prompt minus generated-with-control-prompt, seeds paired. It
             cancels shifts STACK applies whatever the prompt (consistent with Cell-Eval's
             predicted minus predicted control, if the predicted controls are generated).
    "<name>_one" keeps prompt A, query B (median over draws); "<name>" takes the median over
    draws within each order, then over both orders (swapping leaves the joint target unchanged).
    """
    a, b = double.split("+")
    found = {"raw": {}, "query": {}, "diff": {}}
    for prompt, query in [(a, b), (b, a)]:
        p_rows, q_rows = pools[(prompt, "infer")], pools[(query, "infer")]
        q_counts, q_lib = cells.counts[q_rows], cells.library[q_rows]
        observed = mean_log_profile(q_counts, q_lib, cells.genes, panel)
        found["query"][(prompt, query)] = [observed]
        found["raw"][(prompt, query)], found["diff"][(prompt, query)] = [], []
        for draw in range(draws):
            seed = 1000 * draw + 1

            def run(prompt_rows):
                gen = generate(model, cells.counts[prompt_rows], q_counts, cells.genes, genelist_path,
                               num_steps=num_steps, seed=seed, batch_size=batch_size)
                return mean_log_profile(gen, q_lib, cells.genes, panel)

            generated = run(p_rows)
            found["raw"][(prompt, query)].append(generated)
            if reference is not None:
                found["diff"][(prompt, query)].append(observed + generated - run(reference))
    out = {}
    for name, by_order in found.items():
        if by_order[(a, b)]:
            out[f"{name}_one"] = pd.concat(by_order[(a, b)], axis=1).median(axis=1)
            out[name] = median_of_medians(by_order)
    return out


def median_of_medians(profiles: dict) -> pd.Series:
    """Per gene, the median over draws within each transformation, then the median
    over transformations (less sensitive to one poorly behaved transformation)."""
    within = [pd.concat(draws, axis=1).median(axis=1) for draws in profiles.values()]
    return pd.concat(within, axis=1).median(axis=1)
