"""Everything needed to score predictions on one held-out set, saved once and shared.

A dataset script (such as scripts/norman_baselines.py) builds the context: truth
profiles, the control and mean-baseline references, both DE tables, the baseline
predictions and the gene panels. Any model is then scored with `EvalContext.score`,
beside the same baselines, on the same genes. The file is a compressed .npz, loaded
without pickle.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import METRICS, relative_to_negative, score_profiles

DE_COLUMNS = ["score", "pval", "pval_adj"]
LONG_COLUMNS = ["panel", "model", "perturbation", "metric", "value", "gap_closed"]
_PANEL = "panel__"


@dataclass
class EvalContext:
    truth: pd.DataFrame  # held-out perturbations x genes (mean log-normalized expression)
    ctrl: pd.Series  # the Delta-ctrl reference
    pert_ref: pd.Series  # the mean baseline, the Delta-pert reference
    de_vs_ctrl: dict  # perturbation -> DataFrame (score, pval, pval_adj) by gene
    de_vs_rest: dict
    baselines: dict  # name -> predictions; must include control_baseline and mean_baseline
    panels: dict  # panel name -> genes scored on that panel

    def save(self, path) -> None:
        genes, perts = list(self.truth.columns), list(self.truth.index)

        def de_array(tables):
            return np.stack([tables[p].reindex(genes)[DE_COLUMNS].to_numpy(dtype=float) for p in perts])

        np.savez_compressed(
            path,
            genes=np.array(genes, dtype=str),
            perts=np.array(perts, dtype=str),
            truth=self.truth.to_numpy(dtype=float),
            ctrl=self.ctrl[genes].to_numpy(dtype=float),
            pert_ref=self.pert_ref[genes].to_numpy(dtype=float),
            de_vs_ctrl=de_array(self.de_vs_ctrl),
            de_vs_rest=de_array(self.de_vs_rest),
            baseline_names=np.array(list(self.baselines), dtype=str),
            baselines=np.stack([b.loc[perts, genes].to_numpy(dtype=float) for b in self.baselines.values()]),
            **{_PANEL + name: np.array(cols, dtype=str) for name, cols in self.panels.items()},
        )

    @classmethod
    def load(cls, path) -> "EvalContext":
        with np.load(path, allow_pickle=False) as z:
            genes = [str(g) for g in z["genes"]]
            perts = [str(p) for p in z["perts"]]

            def frame(values):
                return pd.DataFrame(values, index=perts, columns=genes)

            def de_tables(values):
                return {p: pd.DataFrame(values[i], index=genes, columns=DE_COLUMNS) for i, p in enumerate(perts)}

            return cls(
                truth=frame(z["truth"]),
                ctrl=pd.Series(z["ctrl"], index=genes),
                pert_ref=pd.Series(z["pert_ref"], index=genes),
                de_vs_ctrl=de_tables(z["de_vs_ctrl"]),
                de_vs_rest=de_tables(z["de_vs_rest"]),
                baselines={str(n): frame(v) for n, v in zip(z["baseline_names"], z["baselines"])},
                panels={k[len(_PANEL):]: [str(g) for g in z[k]] for k in z.files if k.startswith(_PANEL)},
            )

    def score(self, predictions: dict, perturbations=None) -> pd.DataFrame:
        """Long table (panel, model, perturbation, metric, value, gap_closed) for the
        baselines and every model in `predictions` (name -> perturbations x genes).

        `perturbations` restricts scoring to a subset; `nir` then ranks within it.
        """
        perts = list(self.truth.index) if perturbations is None else list(perturbations)
        models = {**self.baselines, **predictions}
        frames = []
        for panel, cols in self.panels.items():
            truth = self.truth.loc[perts, cols]
            scores = {
                name: score_profiles(
                    pred.loc[perts, cols], truth, self.ctrl[cols], self.pert_ref[cols], self.de_vs_ctrl, self.de_vs_rest
                )
                for name, pred in models.items()
            }
            gap = relative_to_negative(scores, mean_baseline="mean_baseline", control_baseline="control_baseline")
            for name in models:
                both = pd.concat({"value": scores[name], "gap_closed": gap[name]}, axis=1)
                for metric in METRICS:
                    rows = both.xs(metric, axis=1, level=1).rename_axis("perturbation").reset_index()
                    frames.append(rows.assign(panel=panel, model=name, metric=metric))
        return pd.concat(frames, ignore_index=True)[LONG_COLUMNS]


def median_tables(long: pd.DataFrame, column: str) -> dict:
    """Per panel, the median of `column` over perturbations as a metric x model table."""
    tables = {}
    for panel, rows in long.groupby("panel", sort=False):
        models = list(dict.fromkeys(rows["model"]))
        medians = rows.groupby(["metric", "model"], sort=False)[column].median().unstack("model")
        tables[panel] = medians.reindex(index=list(METRICS), columns=models).rename_axis("metric")
    return tables


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [f"{v:.3f}" for v in row]) + " |" for i, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])
