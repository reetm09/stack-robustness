"""Leakage and bookkeeping checks for the Experiment 2 splits.

Synthetic tests always run. The `written_*` tests check the saved files in splits/compositional/
and are skipped until `scripts/make_splits.py` has been run.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from stack_compositional.splits import CATEGORIES, all_genes, category, genes_of, s_all_folds, s_gene_split

SPLITS = Path(__file__).resolve().parents[2] / "splits"
OUT = SPLITS / "compositional"
SEEDS = range(5)

GENES = [f"G{i:02d}" for i in range(30)]
PAIRS = [f"{a}+{b}" for a, b in combinations(GENES, 2)][::4]  # 109 sparse pairs
SINGLES = GENES


def test_category_counts_genes_in_training_doubles():
    train = ["A+B", "C+D"]
    assert category("A+C", train) == "seen 2/2"
    assert category("A+E", train) == "unseen 1/2"
    assert category("E+F", train) == "unseen 2/2"


def test_s_all_folds_cover_every_pair_once_and_balance_groups():
    groups = pd.Series(["x"] * 80 + ["y"] * 29, index=PAIRS)
    folds = s_all_folds(PAIRS, groups, k=5, seed=0)
    assert sorted(folds.index) == sorted(PAIRS)
    assert folds.value_counts().max() - folds.value_counts().min() <= 1
    per = pd.crosstab(folds, groups)
    assert (per.max() - per.min()).max() <= 1
    assert folds.equals(s_all_folds(PAIRS, groups, k=5, seed=0))  # deterministic


def _check(test, val):
    """The leakage rules from SPLITS.md, for one seed."""
    doubles = lambda t, s: set(t.loc[(t["kind"] == "combination") & (t["split"] == s), "condition"])
    tr, te = doubles(test, "train"), doubles(test, "test")
    vtr, vv = doubles(val, "train"), doubles(val, "val")
    assert not tr & te and not vtr & vv                   # no pair on both sides
    assert vtr | vv == tr and not (vtr | vv) & te          # validation lives inside training
    held = {g for p in te for g in genes_of(p)} - set(all_genes(tr))
    assert held and not any(set(genes_of(p)) & held for p in tr | vtr | vv)  # held genes never trained on
    for t, train in [(test, tr), (val, vtr)]:
        heldout = t[t["split"].isin(["test", "val"])]
        assert all(heldout["category"] == [category(p, train) for p in heldout["condition"]])
        assert set(heldout["category"]) == set(CATEGORIES)
        assert (t.loc[t["kind"] == "single", "split"] == "train").all()  # singles always available
    return tr, te


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_s_gene_split_has_no_leakage_on_synthetic_pairs(seed):
    test, val, meta = s_gene_split(PAIRS, SINGLES, seed, test_kw={"min_per_category": 2},
                                   val_kw={"min_per_category": 1})
    tr, te = _check(test, val)
    assert tr | te == set(PAIRS)
    assert meta["test"]["tries"] >= 1


def test_s_gene_split_is_deterministic():
    a = s_gene_split(PAIRS, SINGLES, 3, test_kw={"min_per_category": 2}, val_kw={"min_per_category": 1})
    b = s_gene_split(PAIRS, SINGLES, 3, test_kw={"min_per_category": 2}, val_kw={"min_per_category": 1})
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


written = pytest.mark.skipif(not (OUT / "s_gene_meta.json").exists(), reason="run scripts/make_splits.py first")


def _norman():
    cond = pd.read_csv(SPLITS / "norman2019_conditions.csv")
    return (set(cond.loc[cond["kind"] == "combination", "condition"]),
            set(cond.loc[cond["kind"] == "single", "condition"]))


@written
def test_written_s_all_folds():
    doubles, _ = _norman()
    folds = pd.read_csv(OUT / "s_all_folds.csv")
    assert set(folds["condition"]) == doubles and folds["condition"].is_unique
    assert sorted(folds["fold"].unique()) == list(range(5))


@written
@pytest.mark.parametrize("seed", SEEDS)
def test_written_s_gene_seed(seed):
    doubles, singles = _norman()
    test = pd.read_csv(OUT / f"s_gene_seed{seed}_conditions.csv", keep_default_na=False)
    val = pd.read_csv(OUT / f"s_gene_seed{seed}_val_conditions.csv", keep_default_na=False)
    tr, te = _check(test, val)
    assert tr | te == doubles and set(test.loc[test["kind"] == "single", "condition"]) == singles
    meta = json.loads((OUT / "s_gene_meta.json").read_text())[f"seed{seed}"]
    held = {g for p in te for g in genes_of(p)} - set(all_genes(tr))
    assert set(meta["test"]["held_genes"]) <= held        # every drawn gene is fully hidden
    assert held - set(meta["test"]["held_genes"]) == set(meta["test"]["also_unseen"])
    counts = test.loc[test["split"] == "test", "category"].value_counts()
    assert min(counts) >= 5 and len(tr) >= 0.6 * len(doubles)  # the SPLITS.md checks
