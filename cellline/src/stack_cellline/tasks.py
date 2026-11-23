"""Study 1 training and validation windows, built for stack_ttt's adaptation loop.

The proposal's cross-cell-line task, for a knockdown g:
  prompt  cells of g measured in the source line (e.g. HEK293T)
  query   non-targeting control cells of the target line (e.g. HCT116)
  target  cells of g measured in the target line

Training uses the training genes and the target line's training control pool. Validation
uses the validation genes and the validation control pool. Nothing here reads a file:
the caller loads the cells and decides the split. The X-Atlas/Orion loader and the split
reader wait for the team to confirm the data schema and the split format.

stack_ttt.tasks.task_rows takes a task's target rows as every cell whose label equals the
target. Labels are therefore unique per (line, gene) and, for target-line controls, per pool,
and check_split refuses data that hold any cell outside the training and validation sets.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from stack_ttt.norman import Cells
from stack_ttt.tasks import Task

SEP = "|"
ROLE = "adapt"  # the pool key stack_ttt.tasks.task_rows reads prompt and query rows from


@dataclass
class CellLineData:
    """Cells of both lines on STACK's gene vocabulary, already loaded by the caller."""

    counts: sparse.csr_matrix  # cells x genes, raw counts, columns in STACK vocabulary order
    genes: list  # STACK's vocabulary, in column order
    line: np.ndarray  # per cell: its cell line
    target: np.ndarray  # per cell: the knocked-down gene, or the control label


@dataclass
class Split:
    """Which knockdowns and which target-line control cells each stage may use.

    The test genes are listed only so their absence can be checked. Test-pool control
    cells must not be loaded at all: every target-line control cell in the data has to be
    in the training or the validation pool.
    """

    source_line: str
    target_line: str
    control_label: str
    train_genes: list
    val_genes: list
    test_genes: list
    train_controls: np.ndarray  # rows of target-line control cells, for training queries
    val_controls: np.ndarray  # rows of target-line control cells, for validation queries


def check_split(data: CellLineData, split: Split) -> None:
    """Raise ValueError unless training, validation and test are kept apart in `data`."""
    n = data.counts.shape[0]
    if not (len(data.line) == len(data.target) == n):
        raise ValueError("line and target need one entry per cell")
    if data.counts.shape[1] != len(data.genes):
        raise ValueError("counts need one column per vocabulary gene")
    train, val, test = set(split.train_genes), set(split.val_genes), set(split.test_genes)
    if not train or not val:
        raise ValueError("the split needs training and validation genes")
    for a, b, name in [(train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")]:
        if a & b:
            raise ValueError(f"{name} genes overlap: {sorted(a & b)[:5]}")
    if split.control_label in train | val | test:
        raise ValueError("the control label is also listed as a gene")
    if split.source_line == split.target_line:
        raise ValueError("source and target line must differ")
    names = {str(x) for x in np.unique(data.line)} | {str(x) for x in np.unique(data.target)}
    if any(SEP in x for x in names):
        raise ValueError(f"line and gene names must not contain {SEP!r}")

    line = data.line.astype(str)
    target = data.target.astype(str)
    is_ctrl = target == split.control_label
    if not set(line) <= {split.source_line, split.target_line}:
        raise ValueError(f"unexpected lines: {sorted(set(line) - {split.source_line, split.target_line})}")
    loaded_test = set(target[~is_ctrl]) & test
    if loaded_test:
        raise ValueError(f"test genes must not be loaded: {sorted(loaded_test)[:5]}")
    other = set(target[~is_ctrl]) - train - val
    if other:
        raise ValueError(f"cells of genes in neither training nor validation: {sorted(other)[:5]}")

    tr, va = np.asarray(split.train_controls, dtype=int), np.asarray(split.val_controls, dtype=int)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("both control pools need cells")
    if len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va):
        raise ValueError("a control pool lists a row twice")
    if np.intersect1d(tr, va).size:
        raise ValueError("training and validation control pools overlap")
    pooled = np.concatenate([tr, va])
    if pooled.min() < 0 or pooled.max() >= n:
        raise ValueError("control pool rows out of range")
    target_ctrl = np.flatnonzero(is_ctrl & (line == split.target_line))
    if not np.isin(pooled, target_ctrl).all():
        raise ValueError("control pools may hold only target-line control cells")
    unpooled = np.setdiff1d(target_ctrl, pooled)
    if unpooled.size:
        raise ValueError(f"{unpooled.size} target-line control cells are in no pool; "
                         "test-pool controls must not be loaded")

    for gene in sorted(train | val):
        for which in (split.source_line, split.target_line):
            if not np.any((target == gene) & (line == which)):
                raise ValueError(f"{gene} has no cells in {which}")


def _labels(data: CellLineData, split: Split) -> np.ndarray:
    """Per cell, "line|gene"; target-line controls get "line|control|train" or "|val"."""
    labels = np.char.add(np.char.add(data.line.astype(str), SEP), data.target.astype(str)).astype(object)
    ctrl = f"{split.target_line}{SEP}{split.control_label}"
    labels[np.asarray(split.train_controls, dtype=int)] = f"{ctrl}{SEP}train"
    labels[np.asarray(split.val_controls, dtype=int)] = f"{ctrl}{SEP}val"
    return labels.astype(str)


def training_tasks(data: CellLineData, split: Split):
    """(cells, pools, tasks) for stack_ttt.adapt.adapt: one task per training gene."""
    check_split(data, split)
    condition = _labels(data, split)
    library = np.asarray(data.counts.sum(axis=1), dtype=float).ravel()  # not read by adapt()
    cells = Cells(counts=sparse.csr_matrix(data.counts), library=library, condition=condition, genes=list(data.genes))
    query = f"{split.target_line}{SEP}{split.control_label}{SEP}train"
    pools = {(query, ROLE): np.asarray(split.train_controls, dtype=int)}
    tasks = []
    for gene in sorted(split.train_genes):
        prompt = f"{split.source_line}{SEP}{gene}"
        pools[(prompt, ROLE)] = np.flatnonzero(condition == prompt)
        tasks.append(Task(prompt=prompt, query=query, target=f"{split.target_line}{SEP}{gene}"))
    return cells, pools, tasks


def validation_rows(data: CellLineData, split: Split) -> dict:
    """gene -> (prompt rows, query rows, target rows) for every validation gene."""
    check_split(data, split)
    line, target = data.line.astype(str), data.target.astype(str)
    query = np.asarray(split.val_controls, dtype=int)
    return {
        gene: (
            np.flatnonzero((line == split.source_line) & (target == gene)),
            query,
            np.flatnonzero((line == split.target_line) & (target == gene)),
        )
        for gene in sorted(split.val_genes)
    }
