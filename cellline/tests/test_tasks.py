from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from stack_ttt.tasks import task_rows

from stack_cellline.tasks import CellLineData, check_split, training_tasks, validation_rows

from conftest import CONTROL, SOURCE, TARGET, TRAIN_GENES, VAL_GENES


def test_one_cross_line_task_per_training_gene(tiny_data):
    data, split = tiny_data
    cells, pools, tasks = training_tasks(data, split)

    assert [t.target for t in tasks] == [f"{TARGET}|{g}" for g in sorted(TRAIN_GENES)]
    for task in tasks:
        gene = task.target.split("|")[1]
        prompt, query, target = task_rows(task, pools, cells.condition)
        assert set(data.line[prompt]) == {SOURCE} and set(data.target[prompt]) == {gene}
        assert set(query) == set(split.train_controls)
        assert set(data.line[target]) == {TARGET} and set(data.target[target]) == {gene}


def test_training_never_reaches_validation_cells(tiny_data):
    data, split = tiny_data
    cells, pools, tasks = training_tasks(data, split)

    used = set()
    for task in tasks:
        for rows in task_rows(task, pools, cells.condition):
            used |= set(rows.tolist())
    val_cells = set(np.flatnonzero(np.isin(data.target, VAL_GENES)).tolist()) | set(split.val_controls.tolist())
    assert used and not used & val_cells


def test_validation_rows_use_the_validation_pool(tiny_data):
    data, split = tiny_data
    rows = validation_rows(data, split)

    assert sorted(rows) == sorted(VAL_GENES)
    for gene, (prompt, query, target) in rows.items():
        assert set(data.line[prompt]) == {SOURCE} and set(data.target[prompt]) == {gene}
        assert set(query) == set(split.val_controls)
        assert set(data.line[target]) == {TARGET} and set(data.target[target]) == {gene}


def _with_cells(data, line, target):
    """data plus one extra cell of `line` and `target`."""
    extra = sparse.csr_matrix(np.ones((1, data.counts.shape[1]), dtype=np.float32))
    return CellLineData(
        counts=sparse.vstack([data.counts, extra]).tocsr(),
        genes=data.genes,
        line=np.append(data.line, line),
        target=np.append(data.target, target),
    )


def test_refuses_loaded_test_genes(tiny_data):
    data, split = tiny_data
    with pytest.raises(ValueError, match="test genes must not be loaded"):
        check_split(_with_cells(data, TARGET, "G5"), split)


def test_refuses_target_controls_outside_both_pools(tiny_data):
    data, split = tiny_data
    with pytest.raises(ValueError, match="in no pool"):
        check_split(_with_cells(data, TARGET, CONTROL), split)


def test_refuses_genes_outside_the_split(tiny_data):
    data, split = tiny_data
    with pytest.raises(ValueError, match="neither training nor validation"):
        check_split(_with_cells(data, SOURCE, "G9"), split)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: replace(s, val_genes=s.val_genes + ["G0"]), "train/val genes overlap"),
        (lambda s: replace(s, test_genes=["G3"]), "val/test genes overlap"),
        (lambda s: replace(s, val_controls=np.append(s.val_controls, s.train_controls[0])), "pools overlap"),
        (lambda s: replace(s, train_controls=np.append(s.train_controls, 0)), "only target-line control"),
        (lambda s: replace(s, val_genes=[]), "training and validation genes"),
    ],
)
def test_refuses_a_split_that_mixes_stages(tiny_data, change, message):
    data, split = tiny_data
    with pytest.raises(ValueError, match=message):
        check_split(data, change(split))


def test_refuses_a_gene_missing_from_one_line(tiny_data):
    data, split = tiny_data
    keep = ~((data.line == SOURCE) & (data.target == "G1"))
    rows = np.flatnonzero(keep)
    remap = -np.ones(len(keep), dtype=int)
    remap[rows] = np.arange(len(rows))
    dropped = CellLineData(counts=data.counts[rows], genes=data.genes, line=data.line[rows], target=data.target[rows])
    moved = replace(split, train_controls=remap[split.train_controls], val_controls=remap[split.val_controls])
    with pytest.raises(ValueError, match="G1 has no cells in SRC"):
        check_split(dropped, moved)
