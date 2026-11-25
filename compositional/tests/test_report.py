import numpy as np
import pandas as pd
import pytest

from stack_compositional.report import (
    LATEX_COLUMNS, MARKDOWN_COLUMNS, NON_ADDITIVE, latex_table, markdown_table, subset_pairs, summary,
)


def long_table(rows):
    return pd.DataFrame(rows, columns=["panel", "model", "perturbation", "metric", "value"]).assign(gap_closed=np.nan)


@pytest.fixture
def scores():
    pairs = [NON_ADDITIVE[0], NON_ADDITIVE[1], "X+Y", "Z+W"]
    rows = []
    for i, p in enumerate(pairs):
        rows += [
            ("stack_genes", "additive", p, "pearson_deltactrl", 0.8 + 0.01 * i),
            ("stack_genes", "additive", p, "mse", 0.001),
            ("stack_genes", "stack_frozen", p, "pearson_deltactrl", 0.5),
            ("stack_genes", "stack_frozen", p, "mse", 0.002 if p in NON_ADDITIVE else 0.004),
            ("hvg2000", "additive", p, "pearson_deltactrl", 0.0),  # other panel must be ignored
        ]
    return long_table(rows)


def test_subsets_partition_the_pairs(scores):
    perts = scores["perturbation"]
    na, add = subset_pairs(perts, "non-additive"), subset_pairs(perts, "additive")
    assert set(na) | set(add) == set(subset_pairs(perts, "all"))
    assert not set(na) & set(add)
    with pytest.raises(ValueError):
        subset_pairs(perts, "nope")


def test_summary_takes_medians_on_one_panel(scores):
    table = summary(scores, "stack_genes", LATEX_COLUMNS)
    assert table.loc["Additive baseline", "Pearson"] == pytest.approx(0.815)
    assert table.loc["STACK frozen", "MSE"] == pytest.approx(0.003)
    assert np.isnan(table.loc["Ridge regression", "Pearson"])  # not run
    assert np.isnan(table.loc["Additive baseline", "Residual Pearson"])  # metric absent


def test_summary_restricts_to_subset(scores):
    table = summary(scores, "stack_genes", LATEX_COLUMNS, "non-additive")
    assert table.loc["STACK frozen", "MSE"] == pytest.approx(0.002)


def test_tables_mark_missing_and_best(scores):
    table = summary(scores, "stack_genes", MARKDOWN_COLUMNS)
    md = markdown_table(table)
    assert "| Ridge regression | -- | -- | -- | -- |" in md
    assert "**0.815**" in md  # highest Pearson
    assert "**0.00100**" in md  # lowest MSE
    tex = latex_table(summary(scores, "stack_genes", LATEX_COLUMNS))
    assert r"\label{tab:composition_results}" in tex
    assert r"\textbf{0.815}" in tex
    assert tex.count(r"\\") == 7  # header + 6 model rows


def test_empty_scores_give_all_dashes():
    empty = long_table([])
    tex = latex_table(summary(empty, "stack_genes", LATEX_COLUMNS))
    assert "textbf" not in tex
    assert tex.count("--") == 6 * len(LATEX_COLUMNS)


def test_summary_filters_by_truth_mode_and_reads_old_files_as_halves(scores):
    both = pd.concat([scores.assign(truth="halves"),
                      scores.assign(truth="all", value=scores["value"] + 1.0)], ignore_index=True)
    halves = summary(both, "stack_genes", LATEX_COLUMNS, truth="halves")
    allc = summary(both, "stack_genes", LATEX_COLUMNS, truth="all")
    assert allc.loc["Additive baseline", "Pearson"] == pytest.approx(halves.loc["Additive baseline", "Pearson"] + 1)
    legacy = summary(scores, "stack_genes", LATEX_COLUMNS)  # no truth column
    pd.testing.assert_frame_equal(legacy, halves)
    assert summary(scores, "stack_genes", LATEX_COLUMNS, truth="all").isna().all().all()


def test_reference_rows_show_na_only_when_unavailable(scores):
    from stack_compositional.report import REFERENCE_ROWS, ROWS
    rows = ROWS + REFERENCE_ROWS
    table = summary(scores, "stack_genes", MARKDOWN_COLUMNS, rows=rows)
    assert "| Technical duplicate (ref.) | -- |" in markdown_table(table)
    md = markdown_table(table, refs_unavailable=True)
    assert "| Technical duplicate (ref.) | n/a | n/a | n/a | n/a |" in md
    assert "| Ridge regression | -- |" in md  # ordinary rows keep "--"


def test_markdown_extra_column(scores):
    table = summary(scores, "stack_genes", MARKDOWN_COLUMNS)
    extra = pd.Series({"Additive baseline": -0.001})
    md = markdown_table(table, extra=extra, extra_label="wMSE Δ (all − halves)")
    assert "wMSE Δ (all − halves) |" in md.splitlines()[0]
    assert "| -0.0010 |" in md


def test_latex_label_depends_on_truth(scores):
    table = summary(scores, "stack_genes", LATEX_COLUMNS)
    assert r"\label{tab:composition_results}" in latex_table(table)
    assert r"\label{tab:composition_results_allcells}" in latex_table(table, truth="all")
