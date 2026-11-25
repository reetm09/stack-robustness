"""How additive is each measured double perturbation? (no model involved)

In short: for every gene pair A+B we check how well the simple guess "A's effect plus B's
effect" matches the measured double. We place that guess on a ruler that runs from
0 = as good as repeating the experiment (the technical duplicate)
to 1 = as bad as guessing "nothing changes" (control).
We do the same for a stretched/shrunk sum (c1·A + c2·B, a per-pair weight on each gene).
- Near 0 on the plain sum: the pair is additive.
- Far on the plain sum but near 0 on the stretched one: the effects add, but weaker (buffering) or
  stronger (synergy).
- Far on both: a real interaction.

Technical details: definitions follow the pasted reference methods ("metric calibration" and
"specific epistasis model"):
- calibrated score = (d(model) − d(duplicate)) / (d(negative) − d(duplicate)), clipped at 0. It is
  NaN where d(duplicate) >= d(negative), since the calibration is not meaningful there.
- specific epistasis: mu_pq = mu_0 + c1_pq·Delta_p + c2_pq·Delta_q + eps, with pair-specific
  (c1, c2) and one gene-wise offset eps shared by all pairs, from one joint ordinary-least-squares
  fit. The problem is a convex quadratic, solved by alternating exact block updates (per-pair 2×2
  least squares; eps = mean residual) until it converges.
"""

import numpy as np
import pandas as pd


def calibrated(err, err_pos, err_neg):
    """(err − err_pos) / (err_neg − err_pos), clipped at 0; NaN where err_neg <= err_pos."""
    err, err_pos, err_neg = (np.asarray(x, dtype=float) for x in (err, err_pos, err_neg))
    span = err_neg - err_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(span > 0, np.maximum((err - err_pos) / span, 0.0), np.nan)
    return out if out.ndim else float(out)


def specific_epistasis_fit(ctrl: pd.Series, effects: pd.DataFrame, doubles: pd.DataFrame,
                           max_iter: int = 500, tol: float = 1e-10):
    """Joint OLS fit of mu_pq = ctrl + c1·Delta_p + c2·Delta_q + eps over all rows of `doubles`.

    `ctrl`: control mean (genes); `effects`: single-perturbation effects Delta (singles × genes);
    `doubles`: measured means of the doubles (index "A+B", same genes).
    Returns (coefs DataFrame [c1, c2] by pair, eps Series by gene, fitted DataFrame like `doubles`).
    """
    genes = list(doubles.columns)
    pairs = list(doubles.index)
    Y = doubles[genes].to_numpy(dtype=float) - ctrl[genes].to_numpy(dtype=float)  # pairs × genes
    A = np.stack([effects.loc[p.split("+")[0], genes].to_numpy(dtype=float) for p in pairs])
    B = np.stack([effects.loc[p.split("+")[1], genes].to_numpy(dtype=float) for p in pairs])
    eps = np.zeros(len(genes))
    coefs = np.zeros((len(pairs), 2))
    last = np.inf
    for _ in range(max_iter):
        R = Y - eps
        for i in range(len(pairs)):
            M = np.column_stack([A[i], B[i]])
            coefs[i] = np.linalg.lstsq(M, R[i], rcond=None)[0]
        signal = coefs[:, :1] * A + coefs[:, 1:] * B
        eps = (Y - signal).mean(axis=0)
        loss = float(((Y - signal - eps) ** 2).sum())
        if last - loss <= tol * max(1.0, loss):
            break
        last = loss
    fitted = pd.DataFrame(ctrl[genes].to_numpy() + signal + eps, index=pairs, columns=genes)
    return (pd.DataFrame(coefs, index=pairs, columns=["c1", "c2"]), pd.Series(eps, index=genes), fitted)


GROUPS = ("additive", "buffering/synergy", "interaction")


def additivity_groups(plain, stretched, q: float = 0.75):
    """Label each pair additive / buffering/synergy / interaction (agreed rule, step 2.6e).

    - cut = the q-quantile of the plain-sum scores (q = 0.75: the top 25% are non-additive);
    - non-additive: plain-sum score > cut;
      - interaction: the stretched sum is still > cut (even the best per-pair weights leave the
        pair worse than the additive pairs);
      - buffering/synergy: otherwise;
    - additive: everything else.
    Pairs with a missing plain score (redo not better than control) get NaN.
    Returns (labels Series, cut).
    """
    plain = pd.Series(plain, dtype=float)
    stretched = pd.Series(stretched, dtype=float).reindex(plain.index)
    cut = float(plain.quantile(q))
    labels = pd.Series(np.where(plain > cut, np.where(stretched > cut, "interaction", "buffering/synergy"), "additive"),
                       index=plain.index, dtype=object)
    labels[plain.isna()] = np.nan
    return labels, cut


def additivity_scores(adata, halves: pd.Series, panel_genes, de_rest: dict, control: str = "control") -> pd.DataFrame:
    """Ruler scores of every double in `adata` (log-normalized, obs["condition"]) on `panel_genes`.

    Truth = first-half mean, redo = second-half mean, singles and control from all their cells;
    wMSE weights from `de_rest` (DE of each condition's first half vs all other perturbed
    first-half cells, as in the shared scoring). Returns plain / stretched scores and c1, c2.
    """
    from stack_robustness.baselines import pseudobulk
    from stack_robustness.metrics import deg_weights, wmse

    panel_genes = list(panel_genes)
    sub = adata[:, panel_genes]
    cond = sub.obs["condition"]
    half = halves.reindex(sub.obs_names)
    first, second = (half == "first").to_numpy(), (half == "second").to_numpy()
    prof = pseudobulk(sub.X, cond, panel_genes)
    ctrl = prof.loc[control]
    doubles = sorted(c for c in prof.index if "+" in c)
    singles = sorted(c for c in prof.index if c != control and "+" not in c)
    eff = prof.loc[singles] - ctrl
    truth = pseudobulk(sub.X[first], cond[first], panel_genes).loc[doubles]
    redo = pseudobulk(sub.X[second], cond[second], panel_genes).loc[doubles]
    coefs, _, stretched = specific_epistasis_fit(ctrl, eff, truth)
    rows = {}
    for d in doubles:
        a, b = d.split("+")
        w = deg_weights(de_rest[d]["score"].reindex(panel_genes).to_numpy())
        t = truth.loc[d].to_numpy()

        def err(pred):
            return wmse(np.asarray(pred, dtype=float), t, w)

        e_redo, e_none = err(redo.loc[d]), err(ctrl)
        rows[d] = {"plain": calibrated(err(ctrl + eff.loc[a] + eff.loc[b]), e_redo, e_none),
                   "stretched": calibrated(err(stretched.loc[d]), e_redo, e_none),
                   "c1": coefs.loc[d, "c1"], "c2": coefs.loc[d, "c2"]}
    return pd.DataFrame(rows).T.astype(float)
