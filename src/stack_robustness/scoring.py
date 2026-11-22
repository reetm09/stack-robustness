"""Score predicted pseudobulk profiles with the shared metric set.

Inputs are profiles of mean log-normalized expression (perturbations x genes).
`ctrl` is the query controls' mean (the Delta-ctrl reference) and `pert_ref` the
mean baseline (the Delta-pert reference). Two DE results per perturbation, both
computed on the ground-truth cells, feed different metric families:

- DE against control (Cell-Eval style): the true DE genes for the Spearman, PR-AUC,
  direction and top-gene metrics.
- DE against all other cells (Miller et al.): the gene weights for weighted MSE and
  the top-N genes for the "_top" Pearson metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as m
from .calibration import drf


@dataclass(frozen=True)
class MetricInfo:
    perfect: float
    higher_is_better: bool
    negative: str  # negative control for DRF: "mean" or "control" (as in Miller et al.)


METRICS = {
    "pearson_deltactrl": MetricInfo(1.0, True, "mean"),
    "spearman_deltactrl_de": MetricInfo(1.0, True, "mean"),
    "de_pr_auc": MetricInfo(1.0, True, "mean"),
    "direction_agreement": MetricInfo(1.0, True, "mean"),
    "top_overlap": MetricInfo(1.0, True, "mean"),
    "mse": MetricInfo(0.0, False, "mean"),
    "mae": MetricInfo(0.0, False, "mean"),
    "wmse": MetricInfo(0.0, False, "mean"),
    "pearson_deltapert": MetricInfo(1.0, True, "control"),
    "pearson_deltactrl_top": MetricInfo(1.0, True, "mean"),
    "pearson_deltapert_top": MetricInfo(1.0, True, "control"),
    "nir": MetricInfo(1.0, True, "mean"),
}


def _top_mask(pval_adj, tiebreak, k: int) -> np.ndarray:
    """The k genes with the smallest adjusted p-value, ties broken by larger |tiebreak|."""
    order = np.lexsort((-np.abs(tiebreak), np.nan_to_num(pval_adj, nan=1.0)))
    mask = np.zeros(len(pval_adj), dtype=bool)
    mask[order[:k]] = True
    return mask


def score_profiles(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    ctrl: pd.Series,
    pert_ref: pd.Series,
    de_vs_ctrl: dict,
    de_vs_rest: dict,
    top_n: int = 100,
    overlap_k: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per perturbation in `pred`, one column per metric in METRICS."""
    genes = pred.columns
    truth = truth.loc[pred.index, genes]
    c = ctrl[genes].to_numpy(dtype=float)
    r = pert_ref[genes].to_numpy(dtype=float)
    rows = {}
    for p in pred.index:
        pv = pred.loc[p].to_numpy(dtype=float)
        tv = truth.loc[p].to_numpy(dtype=float)
        dc = de_vs_ctrl[p].reindex(genes)
        dr = de_vs_rest[p].reindex(genes)
        sig = (dc["pval_adj"] < alpha).to_numpy()
        top_ctrl = _top_mask(dc["pval_adj"].to_numpy(), tv - c, overlap_k)
        top_rest = _top_mask(dr["pval_adj"].to_numpy(), dr["score"].to_numpy(), top_n)
        rows[p] = {
            "pearson_deltactrl": m.pearson_delta(pv, tv, c),
            "spearman_deltactrl_de": m.spearman_on(pv - c, tv - c, sig),
            "de_pr_auc": m.de_pr_auc(pv - c, sig),
            "direction_agreement": m.direction_agreement(pv - c, tv - c, sig),
            "top_overlap": m.top_k_overlap(pv - c, top_ctrl),
            "mse": m.mse(pv, tv),
            "mae": m.mae(pv, tv),
            "wmse": m.wmse(pv, tv, m.deg_weights(dr["score"].to_numpy())),
            "pearson_deltapert": m.pearson_delta(pv, tv, r),
            "pearson_deltactrl_top": m.pearson_delta(pv[top_rest], tv[top_rest], c[top_rest]),
            "pearson_deltapert_top": m.pearson_delta(pv[top_rest], tv[top_rest], r[top_rest]),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["nir"] = m.nir(pred.to_numpy(dtype=float), truth.to_numpy(dtype=float))
    return out


def relative_to_negative(scores: dict, mean_baseline: str, control_baseline: str) -> dict:
    """For each model, the share of the gap from each metric's negative control to a
    perfect score that the model closes, per perturbation (clipped to [-1, 1]).

    For the interpolated duplicate this is Miller et al.'s dynamic range fraction.
    """
    negatives = {"mean": scores[mean_baseline], "control": scores[control_baseline]}
    out = {}
    for name, table in scores.items():
        columns = {}
        for metric in table.columns:
            info = METRICS[metric]
            neg = negatives[info.negative][metric].reindex(table.index)
            columns[metric] = [
                drf(model, base, info.perfect, info.higher_is_better) for model, base in zip(table[metric], neg)
            ]
        out[name] = pd.DataFrame(columns, index=table.index)
    return out
