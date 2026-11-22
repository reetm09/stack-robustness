import math

import numpy as np
import pytest
from scipy import sparse

from stack_robustness.data import log_normalize, split_conditions


def test_log_normalize_scales_each_cell_by_its_given_library_size():
    counts = sparse.csr_matrix(np.array([[1.0, 3.0], [0.0, 2.0]], dtype=np.float32))
    # Library sizes cover genes outside this matrix, so they exceed the row sums.
    library = np.array([10.0, 8.0])

    out = log_normalize(counts, library).toarray()

    assert out[0] == pytest.approx([math.log1p(1000.0), math.log1p(3000.0)], rel=1e-6)
    assert out[1] == pytest.approx([0.0, math.log1p(2500.0)], rel=1e-6)


CONDITIONS = ["control", "A", "B", "C", "D"] + [f"{a}+{b}" for a, b in
                                                ["AB", "AC", "AD", "BC", "BD", "CD", "BA", "CA", "DA", "DB"]]


def test_split_conditions_holds_out_only_the_requested_share_of_combinations():
    split = split_conditions(CONDITIONS, control="control", seed=0, test_fraction=0.2)

    test = split.loc[split["split"] == "test", "condition"]
    assert len(test) == 2
    assert all("+" in c for c in test)
    assert split.loc[split["kind"] == "single", "split"].eq("train").all()
    assert "control" not in set(split["condition"])
    assert len(split) == len(CONDITIONS) - 1


def test_split_conditions_is_reproducible_for_a_seed():
    first = split_conditions(CONDITIONS, control="control", seed=0, test_fraction=0.2)
    again = split_conditions(list(reversed(CONDITIONS)), control="control", seed=0, test_fraction=0.2)
    other = split_conditions(CONDITIONS, control="control", seed=3, test_fraction=0.2)

    held_out = lambda s: set(s.loc[s["split"] == "test", "condition"])
    assert held_out(first) == held_out(again)
    assert held_out(first) != held_out(other)
