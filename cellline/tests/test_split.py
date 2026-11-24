"""The Study 1 split builder on synthetic cell metadata (no real data)."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from stack_cellline.split import build_split, main, read_cells

LINES = ["A", "B"]
CTRL = "non-targeting"
ELIGIBLE = [f"G{i:02d}" for i in range(20)]


def make_cells() -> pd.DataFrame:
    """Two lines, each with GEM batches b1 and b2 holding controls (40 and 21) and b3 holding none.
    G00..G19: 120 cells per line in b1/b2, two guides each.
    SHORT: 60 per line. ONESIDE: 150 in A only. NOCTRL: 80 per line in b1/b2 plus 80 in b3."""
    rows = []

    def add(line, gene, batch, n, guides=("g1", "g2")):
        for i in range(n):
            rows.append((f"{line}-{gene}-{batch}-{i}", line, gene, batch, f"{gene}_{guides[i % len(guides)]}"))

    for line in LINES:
        for gene in ELIGIBLE:
            add(line, gene, "b1", 70)
            add(line, gene, "b2", 50)
        add(line, "SHORT", "b1", 60)
        add(line, "NOCTRL", "b1", 80)
        add(line, "NOCTRL", "b3", 80)
        add(line, CTRL, "b1", 40, guides=("nt1", "nt2"))
        add(line, CTRL, "b2", 21, guides=("nt1", "nt2"))
    add("A", "ONESIDE", "b1", 150)
    return pd.DataFrame(rows, columns=["cell_id", "line", "gene", "batch", "guide"])


@pytest.fixture
def built():
    return build_split(make_cells(), LINES, CTRL)


def test_eligibility_follows_the_proposal_rule(built):
    genes, _, _ = built
    assert list(genes.index[genes["split"] != "excluded"]) == ELIGIBLE
    assert genes.at["SHORT", "reason"] == "A: 60 < 100; B: 60 < 100"
    assert genes.at["ONESIDE", "reason"] == "B: 0 < 100"
    # cells in batch b3, which has no controls, do not count
    assert genes.at["NOCTRL", "n_cells_A"] == 80 and genes.at["NOCTRL", "split"] == "excluded"
    assert (genes.loc[ELIGIBLE, "n_guides"] == 2).all()


def test_genes_split_70_15_15_by_gene(built):
    genes, _, summary = built
    counts = genes["split"].value_counts()
    assert (counts["train"], counts["val"], counts["test"], counts["excluded"]) == (14, 3, 3, 3)
    assert summary["genes"] == {"targets": 23, "train": 14, "val": 3, "test": 3, "excluded": 3}


def test_controls_split_within_each_batch(built):
    _, controls, summary = built
    cells = make_cells()
    pooled = cells[cells["gene"] == CTRL]
    assert sorted(controls["cell_id"]) == sorted(pooled["cell_id"])  # every control exactly once
    assert set(controls["batch"]) == {"b1", "b2"}
    sizes = controls.groupby(["line", "batch", "pool"]).size()
    for line in LINES:
        assert (sizes[(line, "b1", "train")], sizes[(line, "b1", "val")], sizes[(line, "b1", "test")]) == (28, 6, 6)
        assert (sizes[(line, "b2", "train")], sizes[(line, "b2", "val")], sizes[(line, "b2", "test")]) == (15, 3, 3)
    assert summary["controls"]["A"] == {"batches_with_controls": 2, "batches_without_controls": 1,
                                        "train": 43, "val": 9, "test": 9}


def test_same_seed_gives_the_same_split_whatever_the_row_order(built):
    genes, controls, _ = built
    shuffled = make_cells().sample(frac=1.0, random_state=7)
    genes2, controls2, _ = build_split(shuffled, LINES, CTRL)
    pd.testing.assert_frame_equal(genes, genes2)
    pd.testing.assert_frame_equal(controls, controls2)

    other, _, _ = build_split(make_cells(), LINES, CTRL, seed=1)
    assert not other["split"].equals(genes["split"])


def _without(cells, mask):
    return cells[~mask].reset_index(drop=True)


@pytest.mark.parametrize(
    "change, kwargs, message",
    [
        (lambda c: pd.concat([c, c.iloc[:1]]), {}, "cell ids must be unique"),
        (lambda c: _without(c, (c["line"] == "B") & (c["gene"] == CTRL)), {}, "has no GEM batch with"),
        (lambda c: c, {"fractions": (0.7, 0.2, 0.2)}, "summing to 1"),
        (lambda c: c, {"control_label": "NTC"}, "has no GEM batch with"),
        (lambda c: c, {"min_cells": 1000}, "eligible genes"),
        (lambda c: c, {"lines": ["A", "C"]}, "no cells of line 'C'"),
    ],
)
def test_refuses_inputs_it_cannot_split(change, kwargs, message):
    args = {"lines": LINES, "control_label": CTRL} | kwargs
    with pytest.raises(ValueError, match=message):
        build_split(change(make_cells()), **args)


def _write_h5ad(cells, path):
    obs = cells.set_index("cell_id").rename(columns={"line": "cell_line", "gene": "target", "batch": "gem"})
    obs.index = obs.index.astype(str)
    ad.AnnData(obs=obs).write_h5ad(path)


COMMON = ["--gene-col", "target", "--batch-col", "gem", "--control-label", CTRL, "--lines", "A", "B"]


def test_command_line_writes_the_same_files_every_time(tmp_path, capsys):
    _write_h5ad(make_cells(), tmp_path / "cells.h5ad")
    argv = ["--obs", str(tmp_path / "cells.h5ad"), "--line-col", "cell_line", "--guide-col", "guide", *COMMON]

    main([*argv, "--out", str(tmp_path / "one")])
    main([*argv, "--out", str(tmp_path / "two")])

    for name in ("genes.csv", "controls.csv.gz", "split.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name
    genes = pd.read_csv(tmp_path / "one" / "genes.csv", index_col="gene")
    expected, controls, _ = build_split(make_cells(), LINES, CTRL)
    assert genes["split"].to_dict() == expected["split"].to_dict()
    pooled = pd.read_csv(tmp_path / "one" / "controls.csv.gz")
    assert pooled.astype(str).equals(controls.astype(str))
    summary = json.loads((tmp_path / "one" / "split.json").read_text())
    assert summary["settings"]["fractions"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert summary["settings"]["seed"] == 0 and summary["inputs"]["columns"]["gene_col"] == "target"
    assert "14 train, 3 val, 3 test" in capsys.readouterr().out


def test_refuses_to_overwrite_a_split(tmp_path):
    _write_h5ad(make_cells(), tmp_path / "cells.h5ad")
    argv = ["--obs", str(tmp_path / "cells.h5ad"), "--line-col", "cell_line", *COMMON, "--out", str(tmp_path / "s")]
    main(argv)
    with pytest.raises(FileExistsError, match="--overwrite"):
        main(argv)
    main([*argv, "--overwrite"])


def test_one_file_per_line_gives_the_same_split(tmp_path):
    cells = make_cells()
    for line in LINES:
        part = cells[cells["line"] == line].drop(columns="line")
        part.rename(columns={"gene": "target", "batch": "gem"}).to_csv(tmp_path / f"{line}.csv.gz", index=False)
    read = read_cells([f"{line}={tmp_path / f'{line}.csv.gz'}" for line in LINES], "target", "gem",
                      cell_id_col="cell_id", guide_col="guide")

    genes, controls, _ = build_split(read, LINES, CTRL)
    expected, expected_controls, _ = build_split(cells, LINES, CTRL)
    pd.testing.assert_frame_equal(genes, expected)
    pd.testing.assert_frame_equal(controls, expected_controls)


def test_a_table_needs_a_cell_id_column(tmp_path):
    make_cells().to_csv(tmp_path / "cells.csv", index=False)
    with pytest.raises(ValueError, match="--cell-id-col"):
        read_cells([str(tmp_path / "cells.csv")], "gene", "batch", line_col="line")


def test_query_filters_cells_before_counting(tmp_path):
    cells = make_cells()
    cells["guide_qc"] = np.where((cells["gene"] == "G00") & (cells["line"] == "B") & (cells["batch"] == "b2"),
                                 "fail", "pass")
    cells.to_csv(tmp_path / "cells.csv", index=False)
    read = read_cells([str(tmp_path / "cells.csv")], "gene", "batch", line_col="line", cell_id_col="cell_id",
                      query="guide_qc == 'pass'")

    genes, _, _ = build_split(read, LINES, CTRL)
    assert genes.at["G00", "split"] == "excluded" and genes.at["G00", "reason"] == "B: 70 < 100"
