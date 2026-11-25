"""Scoring predictions against the PBMC contexts, and the robustness summaries.

Scores use stack_robustness's metrics and its gap-to-negative-control (DRF-style) column. The
fixed baselines (control, mean, technical and interpolated duplicate) are scored once.

Robustness summaries (proposal, Robustness): clean performance S(0), degradation
S(0) - S(sigma), the normalised area under performance-vs-severity, and the worst severity.
Intervals come from a bootstrap over drugs (each drug's targets resampled together).
"""

import numpy as np
import pandas as pd

from stack_robustness.scoring import METRICS, relative_to_negative, score_profiles

KEYS = ["group", "panel", "model", "perturbation"]


class Scorer:
    def __init__(self, contexts: dict):
        self.contexts = contexts
        self.base = {}
        for group, ctx in contexts.items():
            for panel, cols in ctx.panels.items():
                self.base[(group, panel)] = {name: self._raw(ctx, pred, cols, list(ctx.truth.index))
                                             for name, pred in ctx.baselines.items()}

    @staticmethod
    def _raw(ctx, pred: pd.DataFrame, cols, ids) -> pd.DataFrame:
        return score_profiles(pred.loc[ids, cols], ctx.truth.loc[ids, cols], ctx.ctrl[cols], ctx.pert_ref[cols],
                              ctx.de_vs_ctrl, ctx.de_vs_rest)

    @staticmethod
    def _long(scores: pd.DataFrame, gap: pd.DataFrame, **keys) -> pd.DataFrame:
        both = pd.concat({"value": scores, "gap_closed": gap}, axis=1).stack(level=1, future_stack=True)
        both.index.names = ["perturbation", "metric"]
        return both.reset_index().assign(**keys)

    def baselines(self) -> pd.DataFrame:
        frames = []
        for (group, panel), scores in self.base.items():
            gap = relative_to_negative(scores, "mean_baseline", "control_baseline")
            frames += [self._long(scores[n], gap[n], group=group, panel=panel, model=n) for n in scores]
        return pd.concat(frames, ignore_index=True)

    def score(self, predictions: dict, panels_of: dict | None = None) -> pd.DataFrame:
        """predictions: model -> DataFrame (target ids x genes). panels_of: model -> panels it is
        scored on (default: all). Targets missing from a prediction are skipped."""
        frames = []
        for group, ctx in self.contexts.items():
            for panel, cols in ctx.panels.items():
                base = self.base[(group, panel)]
                for name, pred in predictions.items():
                    if panels_of and name in panels_of and panel not in panels_of[name]:
                        continue
                    ids = [i for i in ctx.truth.index if i in pred.index]
                    if not ids:
                        continue
                    scores = self._raw(ctx, pred, cols, ids)
                    gap = relative_to_negative({name: scores, "mean_baseline": base["mean_baseline"].loc[ids],
                                                "control_baseline": base["control_baseline"].loc[ids]},
                                               "mean_baseline", "control_baseline")[name]
                    frames.append(self._long(scores, gap, group=group, panel=panel, model=name))
        return pd.concat(frames, ignore_index=True)


def higher_is_better(metric: str) -> bool:
    return METRICS[metric].higher_is_better


def drug_of(target_id: str) -> str:
    return target_id.split("|")[0]


def per_target(scores: pd.DataFrame, metric: str, panel: str, value: str = "value") -> pd.DataFrame:
    """Mean over draws: one row per (arm, condition, severity, target, model, model input)."""
    rows = scores[(scores["metric"] == metric) & (scores["panel"] == panel)]
    keys = ["arm", "condition", "severity", "target", "model", "perturbation"]
    return rows.groupby(keys, dropna=False, observed=True)[value].mean().reset_index()


def cluster_bootstrap(table: pd.DataFrame, column: str, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Mean of `column` over targets and its 95% interval, resampling drugs."""
    drugs = table["perturbation"].map(drug_of).to_numpy()
    values = table[column].to_numpy(dtype=float)
    unique = np.unique(drugs)
    sums = {d: (np.nansum(values[drugs == d]), np.sum(~np.isnan(values[drugs == d]))) for d in unique}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        s, n = map(sum, zip(*(sums[d] for d in pick)))
        boots.append(s / n if n else np.nan)
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    return float(np.nanmean(values)), float(low), float(high)


def curves(scores: pd.DataFrame, metric: str, panel: str, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per arm, target and model: mean score at each severity, with the clean score at severity 0
    for the corruption arms (sample-size arms have no zero point; their largest n is the clean
    PROMPT_CELLS / QUERY_CELLS setting)."""
    t = per_target(scores, metric, panel)
    out = [t[t["arm"] != "clean"]]
    clean_rows = t[t["arm"] == "clean"] if clean is None else clean
    for arm in ("thinning", "ambient", "contamination"):
        for target in t.loc[t["arm"] == arm, "target"].unique():
            out.append(clean_rows.assign(arm=arm, target=target, severity=0.0))
    return pd.concat(out, ignore_index=True)


def summarise_arm(curve: pd.DataFrame, metric: str, n_boot: int = 2000) -> pd.DataFrame:
    """Per (arm, target, model): clean score, degradation at each severity, normalised AUC and
    worst severity; means over targets with drug-bootstrap intervals."""
    sign = 1.0 if higher_is_better(metric) else -1.0
    rows = []
    for (arm, target, model), c in curve.groupby(["arm", "target", "model"], dropna=False):
        wide = c.pivot_table(index="perturbation", columns="severity", values="value")
        sev = np.array(sorted(wide.columns))
        mean_curve = wide[sev].mean()
        row = {"arm": arm, "target": target, "model": model}
        if 0.0 in wide.columns:
            for s in sev[sev > 0]:
                d = (sign * (wide[0.0] - wide[s])).rename("d").reset_index()
                m, lo, hi = cluster_bootstrap(d, "d", n_boot)
                row[f"degradation@{s:g}"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
        if len(sev) > 1:
            row["auc"] = float(np.trapezoid(mean_curve.to_numpy(), sev) / (sev[-1] - sev[0]))
        row["worst"] = float(mean_curve.min() if sign > 0 else mean_curve.max())
        rows.append(row)
    out = pd.DataFrame(rows)
    degradations = sorted((c for c in out.columns if c.startswith("degradation@")), key=lambda c: float(c[12:]))
    return out[["arm", "target", "model", *degradations, *[c for c in ("auc", "worst") if c in out]]]


def plate_gap(scores: pd.DataFrame, metric: str, panel: str, n_boot: int = 2000) -> pd.DataFrame:
    """Covariate reliance Gamma_plate = s(prompt from the query's plate) - s(prompt from the other
    plate), per model, as a mean over targets with a drug-bootstrap interval."""
    t = per_target(scores[scores["arm"] == "plate"], metric, panel)
    wide = t.pivot_table(index=["model", "perturbation"], columns="condition", values="value").reset_index()
    rows = []
    for model, w in wide.groupby("model"):
        d = w.assign(gap=w["plate_same"] - w["plate_other"])
        m, lo, hi = cluster_bootstrap(d, "gap", n_boot)
        rows.append({"model": model, "gamma_plate": m, "ci_low": lo, "ci_high": hi, "targets": len(d)})
    return pd.DataFrame(rows)
