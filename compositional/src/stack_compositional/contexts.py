"""Scoring contexts for Experiment 2, in two truth modes.

In short: to grade a prediction we need the "right answer" for each held-out gene pair.
There are two ways to build it:

- **halves** (team convention, primary): each pair's cells are split 50/50 at random. The first
  half is the right answer. The second half is a *technical duplicate*: a repeat measurement used
  as the best-achievable reference.
- **all** (extra robustness view): the right answer uses *all* of the pair's cells. This is less
  noisy, which matters most for small pairs. Nothing is left over for a duplicate, so the
  duplicate references are absent.

Every model is scored both ways with the same metrics.

Technical details: this mirrors `scripts/norman_baselines.py` (same baselines, DE tests and
panels) but takes the held-out set and the truth mode as arguments and returns an
`EvalContext`.
- halves: truth = mean of first-half cells; DE (vs rest, t-test) on first-half perturbed cells;
  DE (vs control, Wilcoxon) on first-half held-out cells + all control cells; baselines include
  `technical_duplicate` and `interpolated_duplicate` (DE of second halves vs all other cells).
- all: the same with every cell, and without the duplicate baselines.

Model inputs (singles, controls) are unaffected by the mode.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from stack_robustness.baselines import additive, interpolated_duplicate, mean_baseline, pseudobulk
from stack_robustness.data import log_normalize
from stack_robustness.de import rank_genes
from stack_robustness.evaluation import EvalContext
from stack_robustness.genes import load_stack_genes

CONTROL = "control"
TRUTH_MODES = ("halves", "all")


def load_norman(raw_path, vocab_path, min_cells: int = 3) -> ad.AnnData:
    """Log-normalized STACK-vocabulary genes detected in >= min_cells cells; obs has `condition`.

    Same preprocessing as `scripts/norman_baselines.load_log_normalized` (library = `ncounts`).
    """
    backed = ad.read_h5ad(raw_path, backed="r")
    in_vocab = backed.var_names.isin(set(load_stack_genes(vocab_path)))
    X = sparse.csc_matrix(backed.X[:, in_vocab])  # stored column-major: reads only these genes
    obs = backed.obs[["perturbation", "gemgroup"]].copy()
    library = backed.obs["ncounts"].to_numpy()
    genes = backed.var_names[in_vocab]
    backed.file.close()
    detected = np.diff(X.indptr) >= min_cells
    X = log_normalize(X[:, detected].tocsr(), library)
    obs["condition"] = obs["perturbation"].astype(str).str.replace("_", "+", regex=False)
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes[detected]))


def build_context(adata: ad.AnnData, test, train, halves: pd.Series, truth: str = "halves",
                  n_hvg: int = 2000) -> EvalContext:
    """EvalContext for the held-out conditions `test`, scored in truth mode `truth`.

    `adata`: log-normalized, obs["condition"]; `train`: training conditions (for the mean
    baseline and the HVG panel); `halves`: "first"/"second" per cell (index = obs_names).
    """
    if truth not in TRUTH_MODES:
        raise ValueError(f"truth must be one of {TRUTH_MODES}, got {truth!r}")
    test, train = list(test), list(train)
    genes = list(adata.var_names)
    cond = adata.obs["condition"]
    half = halves.reindex(adata.obs_names)
    first = (half == "first").to_numpy()
    second = (half == "second").to_numpy()
    is_ctrl = (cond == CONTROL).to_numpy()
    in_test = cond.isin(test).to_numpy()
    scored = first if truth == "halves" else np.ones(adata.n_obs, dtype=bool)

    profiles = pseudobulk(adata.X, cond, genes)
    ctrl = profiles.loc[CONTROL]
    mean = mean_baseline(profiles, train)
    truth_profiles = pseudobulk(adata.X[scored], cond[scored], genes).loc[test]

    de_rest = rank_genes(adata[scored & ~is_ctrl].copy(), "condition", reference="rest")
    de_ctrl = rank_genes(adata[(in_test & scored) | is_ctrl].copy(), "condition",
                         reference=CONTROL, method="wilcoxon")

    baselines = {
        "control_baseline": pd.DataFrame([ctrl] * len(test), index=test),
        "mean_baseline": pd.DataFrame([mean] * len(test), index=test),
        "additive": pd.DataFrame({t: additive(ctrl, profiles, t.split("+")) for t in test}).T,
    }
    if truth == "halves":
        duplicate = pseudobulk(adata.X[second], cond[second], genes).loc[test]
        de_dup = rank_genes(adata[second].copy(), "condition", reference="rest")
        baselines["technical_duplicate"] = duplicate
        baselines["interpolated_duplicate"] = pd.DataFrame(
            {t: interpolated_duplicate(duplicate.loc[t], mean, de_dup[t]["pval_adj"]) for t in test}).T

    train_and_ctrl = cond.isin(train).to_numpy() | is_ctrl
    hv = sc.pp.highly_variable_genes(adata[train_and_ctrl].copy(), n_top_genes=n_hvg, inplace=False)
    panels = {"stack_genes": genes, f"hvg{n_hvg}": list(adata.var_names[hv["highly_variable"].to_numpy()])}

    return EvalContext(
        truth=truth_profiles, ctrl=ctrl, pert_ref=mean,
        de_vs_ctrl={t: de_ctrl[t] for t in test}, de_vs_rest={t: de_rest[t] for t in test},
        baselines=baselines, panels=panels,
    )
