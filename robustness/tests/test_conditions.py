import numpy as np

from stack_quality.conditions import Condition, default_conditions, sample_inputs
from stack_quality.pbmc import DMSO


def test_default_conditions_are_unique_and_cover_every_arm():
    conds = default_conditions()
    assert len({c.name for c in conds}) == len(conds)
    assert {c.arm for c in conds} == {"clean", "prompt_size", "query_size", "thinning", "ambient",
                                      "contamination", "plate"}
    assert all(c.corrupts("prompt") for c in conds if c.arm == "thinning" and c.target in ("prompt", "both"))
    assert not any(c.corrupts("query") for c in conds if c.target == "prompt")


def test_query_and_controls_are_dmso_from_the_plate(cells):
    inp = sample_inputs(cells, Condition("clean", "clean", n_prompt=20, n_query=10), 0, "plate_1", ["A", "B"])
    assert (cells.drug[inp.query] == DMSO).all() and (cells.plate[inp.query] == "plate_1").all()
    assert list(inp.query_kind) == ["B"] * 10 + ["My"] * 10
    assert (cells.kind[inp.query] == inp.query_kind).all()
    assert (cells.drug[inp.control_prompt] == DMSO).all() and (cells.kind[inp.control_prompt] == "T").all()
    for drug, rows in inp.prompts.items():
        assert (cells.drug[rows] == drug).all() and (cells.kind[rows] == "T").all()


def test_smaller_sets_are_prefixes_of_larger_ones(cells):
    small = sample_inputs(cells, Condition("s", "prompt_size", n_prompt=8, n_query=5), 1, "plate_1", ["A"])
    big = sample_inputs(cells, Condition("b", "prompt_size", n_prompt=30, n_query=20), 1, "plate_1", ["A"])
    np.testing.assert_array_equal(small.prompts["A"], big.prompts["A"][:8])
    np.testing.assert_array_equal(small.control_prompt, big.control_prompt[:8])
    np.testing.assert_array_equal(small.query[:5], big.query[:5])


def test_contamination_replaces_the_stated_share_with_other_drugs(cells):
    inp = sample_inputs(cells, Condition("c", "contamination", contamination=0.25, n_prompt=40), 0, "plate_1", ["A"])
    rows = inp.prompts["A"]
    assert (cells.drug[rows] != "A").sum() == 10
    assert set(cells.drug[rows]) <= {"A", "B"} and (cells.kind[rows] == "T").all()


def test_plate_swap_takes_prompts_from_the_other_plate(cells):
    inp = sample_inputs(cells, Condition("p", "plate", prompt_plate="other", n_prompt=20), 0, "plate_1", ["A"])
    assert (cells.plate[inp.prompts["A"]] == "plate_2").all()
    assert (cells.plate[inp.control_prompt] == "plate_2").all()
    assert (cells.plate[inp.query] == "plate_1").all()
