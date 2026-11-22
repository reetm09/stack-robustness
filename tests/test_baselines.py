import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from stack_robustness.baselines import (
    additive,
    assign_halves,
    interpolated_duplicate,
    mean_baseline,
    pseudobulk,
    source_effect_copy,
)

GENES = ["g0", "g1"]


def test_pseudobulk_averages_cells_within_each_group_on_sparse_input():
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]))
    labels = pd.Series(["a", "a", "b"])

    out = pseudobulk(X, labels, genes=GENES)

    assert out.loc["a"].tolist() == [2.0, 1.0]
    assert out.loc["b"].tolist() == [5.0, 4.0]
    assert out.columns.tolist() == GENES


def test_pseudobulk_matches_rows_to_labels_by_position_not_by_label_index():
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]))
    labels = pd.Series(["b", "a", "b"], index=["cellZ", "cellA", "cellM"])

    out = pseudobulk(X, labels, genes=GENES)

    assert out.loc["a"].tolist() == [3.0, 2.0]
    assert out.loc["b"].tolist() == [3.0, 2.0]


def test_mean_baseline_averages_only_the_training_perturbations():
    profiles = pd.DataFrame(
        [[0.0, 0.0], [3.0, 3.0], [30.0, 60.0], [100.0, 100.0], [50.0, 50.0]],
        index=["A", "B", "C", "TEST", "control"],
        columns=GENES,
    )

    out = mean_baseline(profiles, train=["A", "B", "C"])

    assert out.tolist() == [11.0, 21.0]


def test_assign_halves_splits_each_group_separately_and_reproducibly():
    labels = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"], index=[f"cell{i}" for i in range(10)])

    halves = assign_halves(labels, seed=0)

    counts = halves.groupby(labels).value_counts()
    assert counts[("a", "first")] == 2 and counts[("a", "second")] == 3
    assert counts[("b", "first")] == 2 and counts[("b", "second")] == 2
    assert pd.isna(halves["cell9"])  # a one-cell group cannot be split
    assert halves.equals(assign_halves(labels, seed=0))
    assert not halves.equals(assign_halves(labels, seed=1))


def test_interpolated_duplicate_weights_duplicate_by_one_minus_adjusted_pvalue():
    dup = pd.Series([10.0, 10.0, 10.0, 10.0])
    mean = pd.Series([0.0, 0.0, 0.0, 0.0])
    pvals_adj = pd.Series([0.0, 1.0, 0.25, np.nan])

    out = interpolated_duplicate(dup, mean, pvals_adj)

    assert out.tolist() == [10.0, 0.0, 7.5, 0.0]


def test_interpolated_duplicate_lines_up_genes_by_name():
    dup = pd.Series([10.0, 10.0], index=["a", "b"])
    mean = pd.Series([5.0, 7.0], index=["b", "a"])
    pvals_adj = pd.Series([1.0, 0.0], index=["b", "a"])

    out = interpolated_duplicate(dup, mean, pvals_adj)

    assert out.to_dict() == {"a": 10.0, "b": 5.0}


def test_additive_adds_single_effects_relative_to_control():
    ctrl = pd.Series([1.0, 1.0], index=GENES)
    singles = pd.DataFrame([[2.0, 1.0], [1.0, 4.0]], index=["A", "B"], columns=GENES)

    out = additive(ctrl, singles, ["A", "B"])

    assert out.tolist() == [2.0, 4.0]


def test_additive_refuses_a_combination_with_an_unmeasured_single():
    ctrl = pd.Series([1.0, 1.0], index=GENES)
    singles = pd.DataFrame([[2.0, 1.0]], index=["A"], columns=GENES)

    with pytest.raises(KeyError):
        additive(ctrl, singles, ["A", "B"])


def test_source_effect_copy_moves_the_prompt_effect_onto_the_query_controls():
    prompt_pert = pd.Series([3.0, 1.0], index=GENES)
    prompt_ctrl = pd.Series([1.0, 1.0], index=GENES)
    query_ctrl = pd.Series([5.0, 5.0], index=GENES)

    out = source_effect_copy(prompt_pert, prompt_ctrl, query_ctrl)

    assert out.tolist() == [7.0, 5.0]
