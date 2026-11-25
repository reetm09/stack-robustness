"""Build the Experiment 2 results table from a long scores table.

In short: every run writes one big table of scores (one row per model, gene pair and
metric). This module boils it down to the small summary table for the paper: one row per
model, one column per metric, each cell the median over the held-out gene pairs. Models
that have not been run yet show "--".

Technical details: the input is the long format written by
`stack_robustness.evaluation.EvalContext.score` (columns panel, model, perturbation,
metric, value, gap_closed) plus a `truth` column: "halves" (scored against the first half of
each pair's cells, team convention) or "all" (scored against all its cells). Files without
the column are read as "halves". Experiment 2 appends its own metric `residual_pearson` in the
same format. Values are medians over perturbations on one gene panel and one truth mode,
optionally restricted to a subset of pairs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Experiment 2's additivity groups (agreed rule, SPLITS.md): written by
# scripts/make_additivity_groups.py; one row per double with group additive / buffering/synergy /
# interaction.
GROUPS_PATH = Path(__file__).resolve().parents[3] / "splits/compositional/additivity_groups.csv"

# For comparison only: Study 4's 9 "non-additive" test doubles (largest additive-baseline wMSE,
# stack_genes panel), copied from ttt/scripts/analyze_ttt.py so this study does not import ttt.
STUDY4_NON_ADDITIVE = [
    "DUSP9+SNAI1", "DUSP9+PRTG", "JUN+CEBPA", "AHR+FEV", "KLF1+COL2A1",
    "CBL+PTPN12", "BPGM+SAMD1", "MAP2K6+SPI1", "AHR+KLF1",
]

# (row label in the table, model name in scores.csv)
ROWS = [
    ("Additive baseline", "additive"),
    ("Weighted prediction", "weighted"),
    ("Ridge regression", "ridge"),
    ("STACK frozen", "stack_frozen"),
    ("STACK fine-tuned", "stack_finetuned"),
    ("STACK, both orders", "stack_both"),
]
# Positive controls, shown in the Markdown tables as reference lines (they exist only in
# halves mode: with all cells used as truth there is no second half left to act as a duplicate).
REFERENCE_ROWS = [
    ("Technical duplicate (ref.)", "technical_duplicate"),
    ("Interpolated duplicate (ref.)", "interpolated_duplicate"),
]
TRUTH_MODES = ("halves", "all")


@dataclass(frozen=True)
class Column:
    label: str  # header in the table
    metric: str  # metric name in scores.csv
    higher_is_better: bool
    fmt: str


LATEX_COLUMNS = [
    Column("Pearson", "pearson_deltactrl", True, "{:.3f}"),
    Column("MSE", "mse", False, "{:.5f}"),
    Column("Residual Pearson", "residual_pearson", True, "{:.3f}"),
]
MARKDOWN_COLUMNS = LATEX_COLUMNS[:2] + [Column("wMSE", "wmse", False, "{:.4f}")] + LATEX_COLUMNS[2:]

SUBSETS = ("all", "non-additive", "additive", "buffering/synergy", "interaction")


def load_groups(path=GROUPS_PATH) -> pd.Series:
    """pair -> additivity group, from splits/compositional/additivity_groups.csv."""
    return pd.read_csv(path, index_col="pair")["group"]


def subset_pairs(perturbations, subset: str, groups: pd.Series | None = None) -> list:
    """The held-out pairs in `subset`.

    "all", or one of the additivity groups ("additive", "buffering/synergy", "interaction"), or
    "non-additive" (= buffering/synergy + interaction). Groups other than "all" need `groups`.
    """
    perts = sorted(set(perturbations))
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    if subset == "all":
        return perts
    if groups is None:
        raise ValueError("subset by additivity needs `groups` (see load_groups)")
    g = groups.reindex(perts)
    keep = g.ne("additive") & g.notna() if subset == "non-additive" else g.eq(subset)
    return [p for p, k in zip(perts, keep) if k]


def with_truth(long: pd.DataFrame) -> pd.DataFrame:
    """The long table with a `truth` column (files from before the column existed are "halves")."""
    return long if "truth" in long.columns else long.assign(truth="halves")


def summary(long: pd.DataFrame, panel: str, columns, subset: str = "all", truth: str = "halves",
            rows=ROWS, pairs=None, groups: pd.Series | None = None) -> pd.DataFrame:
    """Median value per (table row, column) over the pairs in `subset`; NaN where missing.

    `pairs` (optional) fixes the pair list, so tables in both truth modes cover the same pairs.
    """
    long = with_truth(long)
    sel = long[(long["panel"] == panel) & (long["truth"] == truth)]
    keep = subset_pairs(sel["perturbation"] if pairs is None else pairs, subset, groups)
    sel = sel[sel["perturbation"].isin(keep)]
    medians = sel.groupby(["model", "metric"])["value"].median()
    table = pd.DataFrame(index=[label for label, _ in rows], columns=[c.label for c in columns], dtype=float)
    for label, model in rows:
        for c in columns:
            table.loc[label, c.label] = medians.get((model, c.metric), np.nan)
    return table


def _best(table: pd.DataFrame, columns) -> dict:
    """Row label of the best value in each column (ignoring missing values)."""
    best = {}
    for c in columns:
        values = table[c.label].dropna()
        if len(values):
            best[c.label] = values.idxmax() if c.higher_is_better else values.idxmin()
    return best


def _cells(table: pd.DataFrame, columns, bold, not_ranked=(), unavailable=()) -> dict:
    best = _best(table.drop(index=[r for r in not_ranked if r in table.index]), columns)
    out = {}
    for row in table.index:
        out[row] = []
        for c in columns:
            v = table.loc[row, c.label]
            text = ("n/a" if row in unavailable else "--") if pd.isna(v) else c.fmt.format(v)
            if best.get(c.label) == row:
                text = bold(text)
            out[row].append(text)
    return out


def latex_table(table: pd.DataFrame, columns=LATEX_COLUMNS, truth: str = "halves") -> str:
    """The paper's table (label tab:composition_results, or ..._allcells); best value per column in bold."""
    cells = _cells(table, columns, lambda t: rf"\textbf{{{t}}}")
    label = "tab:composition_results" + ("" if truth == "halves" else "_allcells")
    truth_text = ("Scored against a random half of each pair's cells (the other half is the technical "
                  "duplicate)." if truth == "halves" else "Scored against all cells of each pair.")
    header = " & ".join(["Model"] + [c.label for c in columns]) + r" \\"
    body = "\n".join(f"        {row:<22}& " + " & ".join(vals) + r" \\" for row, vals in cells.items())
    return "\n".join([
        r"\begin{table}[t]",
        r"    \caption{Combinatorial perturbation prediction. Median over held-out double",
        r"    perturbations. Pearson: correlation of predicted and true change from control",
        r"    (higher is better). MSE: mean squared error (lower is better). Residual Pearson:",
        r"    correlation of predicted and true non-additive component $r_{AB}$ on each pair's",
        r"    top 100 DE genes (higher is better; undefined for the additive baseline).",
        f"    {truth_text}}}",
        rf"    \label{{{label}}}",
        r"    \centering",
        r"    \begin{tabular}{l" + "c" * len(columns) + "}",
        r"        \toprule",
        "        " + header,
        r"        \midrule",
        body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def markdown_table(table: pd.DataFrame, columns=MARKDOWN_COLUMNS, extra: pd.Series | None = None,
                   extra_label: str = "", extra_fmt: str = "{:+.4f}", refs_unavailable: bool = False) -> str:
    """Markdown table; reference rows are never bolded.

    `refs_unavailable` (all-cells mode) prints "n/a" for the reference rows, which cannot exist
    there. `extra` adds one more column (e.g. the change in wMSE between truth modes), indexed by row.
    """
    refs = [label for label, _ in REFERENCE_ROWS]
    cells = _cells(table, columns, lambda t: f"**{t}**", not_ranked=refs,
                   unavailable=refs if refs_unavailable else ())
    if extra is not None:
        columns = list(columns) + [Column(extra_label, "", False, extra_fmt)]
        for row in cells:
            v = extra.get(row, np.nan)
            cells[row].append("--" if pd.isna(v) else extra_fmt.format(v))
    lines = ["| Model | " + " | ".join(c.label for c in columns) + " |",
             "|---|" + "---|" * len(columns)]
    lines += [f"| {row} | " + " | ".join(vals) + " |" for row, vals in cells.items()]
    return "\n".join(lines)
