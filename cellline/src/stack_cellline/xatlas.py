"""X-Atlas/Orion cells for Study 1, on STACK's gene vocabulary.

The release (Hugging Face `Xaira-Therapeutics/X-Atlas-Orion`, checked 2026-11-24) has one
parquet file per GEM batch, `data/<LINE>_Batch<N>.parquet`, each a single row group with one
row per cell. A cell's nonzero genes are `gene_token_id` (list<int64>) with their raw counts in
`gene_expression` (list<double>); `total_counts` is its library size over all genes.
`metadata/gene_metadata.parquet` maps `gene_token_id` (0..n-1) to `ensembl_id` and `gene_name`.

All of the training cells would need about 130 GB of memory, so loading has two steps:
1. plan_cells picks cells from the per-cell metadata (scripts/fetch_xatlas_obs.py writes one
   table per line) and the split (splits/xatlas_orion): training and validation genes in both
   lines, and the target line's training and validation control pools. It never picks a test
   gene or a test-pool control. `Subset` caps genes, cells and batch files for small runs.
2. load_counts reads only the batch files that hold planned cells.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from .split import _rng
from .split import read_cells as read_metadata
from .tasks import CellLineData, Split, check_split

READ = ["cell_barcode", "sample", "total_counts", "gene_token_id", "gene_expression"]
HF_PREFIX = "hf://"


@dataclass(frozen=True)
class Subset:
    """Caps for a run smaller than the whole split. None means no cap. Each choice is seeded
    on its own, so changing one cap does not change the other choices."""

    n_train_genes: Optional[int] = None
    n_val_genes: Optional[int] = None
    max_cells_per_gene: Optional[int] = None  # per gene and line
    max_controls: Optional[int] = None  # per control pool
    batches_per_line: Optional[int] = None  # batch files read per line
    seed: int = 0


def vocabulary_columns(gene_metadata: pd.DataFrame, vocab, aliases=None) -> np.ndarray:
    """For each gene token, its column in STACK's vocabulary, or -1.

    Symbols match case-insensitively after renaming with `aliases` (data symbol -> vocabulary
    symbol, from stack_robustness.genes.vocabulary_aliases), as stack_ttt.norman.load_norman
    does. Tokens with the same symbol share a column, so their counts add up.
    """
    tokens = gene_metadata["gene_token_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(tokens), np.arange(len(tokens))):
        raise ValueError("gene_token_id must number the genes 0..n-1")
    position = {str(g).upper(): j for j, g in enumerate(vocab)}
    aliases = aliases or {}
    columns = np.full(len(tokens), -1, dtype=np.int64)
    for token, name in zip(tokens, gene_metadata["gene_name"].astype(str)):
        columns[token] = position.get(aliases.get(name, name).upper(), -1)
    return columns


def _choose(items, n: Optional[int], rng) -> list:
    """Up to n of the items, independent of their order."""
    items = sorted(items)
    if n is None or len(items) <= n:
        return items
    return sorted(items[i] for i in rng.choice(len(items), size=n, replace=False))


def _cap(rows: pd.DataFrame, n: Optional[int], rng) -> pd.DataFrame:
    """Up to n of the rows, independent of their order."""
    rows = rows.sort_values("cell_id", kind="mergesort")
    if n is None or len(rows) <= n:
        return rows
    return rows.iloc[np.sort(rng.choice(len(rows), size=n, replace=False))]


def plan_cells(obs: pd.DataFrame, genes: pd.DataFrame, controls: pd.DataFrame, source_line: str,
               target_line: str, control_label: str, subset: Subset = Subset()) -> tuple:
    """(plan, report). `obs` has columns cell_id, line, gene, batch (split.read_cells);
    `genes` and `controls` are the split's genes.csv and controls.csv.gz.

    The plan has one row per cell to load: cell_id, line, gene, batch, and pool for controls.
    Genes left without cells in either line by the caps are dropped and reported.
    """
    seed = subset.seed
    lines = [source_line, target_line]
    obs = obs[obs["line"].isin(lines)]
    batches = {line: _choose(set(obs.loc[obs["line"] == line, "batch"]), subset.batches_per_line,
                             _rng(seed, "batches", line)) for line in lines}
    split_of = genes["split"]
    chosen = {stage: _choose(split_of.index[split_of == stage], n, _rng(seed, "genes", stage))
              for stage, n in (("train", subset.n_train_genes), ("val", subset.n_val_genes))}

    wanted = obs[obs["gene"].isin(set(chosen["train"]) | set(chosen["val"]))]
    wanted = wanted[[row_batch in batches[row_line] for row_line, row_batch in zip(wanted["line"], wanted["batch"])]]
    perturbed = [_cap(group, subset.max_cells_per_gene, _rng(seed, "cells", line, gene))
                 for (line, gene), group in wanted.groupby(["line", "gene"], sort=True)]
    perturbed = pd.concat(perturbed) if perturbed else wanted.iloc[:0]
    have = perturbed.groupby("gene")["line"].nunique()
    complete = set(have.index[have == len(lines)])
    dropped = {stage: sorted(set(g) - complete) for stage, g in chosen.items()}
    chosen = {stage: [g for g in g_list if g in complete] for stage, g_list in chosen.items()}
    perturbed = perturbed[perturbed["gene"].isin(complete)]

    pooled = controls[(controls["line"] == target_line) & controls["pool"].isin(["train", "val"])
                      & controls["batch"].isin(batches[target_line])]
    if set(pooled["pool"]) != {"train", "val"}:
        raise ValueError("the chosen batches hold no training or no validation control cells")
    pooled = pd.concat([_cap(group, subset.max_controls, _rng(seed, "controls", pool))
                        for pool, group in pooled.groupby("pool", sort=True)]).assign(gene=control_label)

    columns = ["cell_id", "line", "gene", "batch", "pool"]
    plan = pd.concat([perturbed.assign(pool="")[columns], pooled[columns]], ignore_index=True)
    test_genes = set(split_of.index[split_of == "test"])
    if set(plan["gene"]) & test_genes:
        raise AssertionError("a test gene was planned")
    if not set(plan.loc[plan["pool"] != "", "pool"]) <= {"train", "val"}:
        raise AssertionError("a test-pool control was planned")
    if plan["cell_id"].duplicated().any():
        raise AssertionError("a cell was planned twice")

    report = {
        "batches": {line: len(b) for line, b in batches.items()},
        "genes": {stage: len(g) for stage, g in chosen.items()},
        "genes_dropped_for_lack_of_cells": {stage: len(g) for stage, g in dropped.items()},
        "cells": {
            "perturbed": {line: int((perturbed["line"] == line).sum()) for line in lines},
            "controls": {pool: int((pooled["pool"] == pool).sum()) for pool in ("train", "val")},
        },
        "train_genes": chosen["train"],
        "val_genes": chosen["val"],
    }
    return plan, report


def _open(data: str, batch: str):
    """A readable file for one batch: in a local directory, or hf://<repo path> on Hugging Face."""
    name = f"{batch}.parquet"
    if data.startswith(HF_PREFIX):
        from huggingface_hub import HfFileSystem

        return HfFileSystem().open(f"{data[len(HF_PREFIX):].rstrip('/')}/{name}", block_size=8 * 2**20)
    return open(Path(data) / name, "rb")


def _to_vocabulary(table: pa.Table, columns: np.ndarray, n_genes: int) -> sparse.csr_matrix:
    """The cells of `table` as a CSR matrix over STACK's vocabulary; tokens outside it are dropped."""
    tokens_list = table["gene_token_id"].combine_chunks()
    values_list = table["gene_expression"].combine_chunks()
    lengths = pc.list_value_length(tokens_list).fill_null(0).to_numpy(zero_copy_only=False)
    if not np.array_equal(lengths, pc.list_value_length(values_list).fill_null(0).to_numpy(zero_copy_only=False)):
        raise ValueError("gene_token_id and gene_expression differ in length")
    tokens = pc.list_flatten(tokens_list).to_numpy(zero_copy_only=False)
    values = pc.list_flatten(values_list).to_numpy(zero_copy_only=False)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    cols = columns[tokens]
    keep = cols >= 0
    return sparse.csr_matrix((values[keep].astype(np.float32), (rows[keep], cols[keep])),
                             shape=(len(lengths), n_genes), dtype=np.float32)


def load_counts(plan: pd.DataFrame, data: str, columns: np.ndarray, n_genes: int, log=print) -> tuple:
    """(counts, library) for the planned cells, in plan order: counts is a cells x vocabulary
    CSR matrix of raw counts, library each cell's `total_counts` over all genes."""
    blocks, libraries, barcodes = [], [], []
    batches = sorted(set(plan["batch"]))
    for i, batch in enumerate(batches, 1):
        wanted = plan.loc[plan["batch"] == batch, "cell_id"].to_numpy(dtype=str)
        with _open(data, batch) as fh:
            table = pq.read_table(fh, columns=READ)
        if set(pc.unique(table["sample"]).to_pylist()) != {batch}:
            raise ValueError(f"{batch}.parquet holds cells of other samples")
        table = table.filter(pc.is_in(table["cell_barcode"], value_set=pa.array(wanted)))
        if table.num_rows != len(wanted):
            raise ValueError(f"{batch}.parquet: found {table.num_rows} of {len(wanted)} planned cells")
        blocks.append(_to_vocabulary(table, columns, n_genes))
        libraries.append(table["total_counts"].to_numpy())
        barcodes += table["cell_barcode"].to_pylist()
        log(f"[cellline] read {batch} ({i}/{len(batches)}): {table.num_rows} cells")
    counts = sparse.vstack(blocks).tocsr()
    library = np.concatenate(libraries)
    rows = pd.Series(np.arange(len(barcodes)), index=barcodes).loc[plan["cell_id"].to_numpy()].to_numpy()
    return counts[rows], library[rows]


def load_xatlas(obs_sources, split_dir, data: str, gene_metadata: pd.DataFrame, vocab, source_line: str,
                target_line: str, subset: Subset = Subset(), aliases=None, plan_only=False, log=print) -> dict:
    """{"data", "split", "library", "plan", "report"}: the cells and split finetune() takes, each
    cell's library size, which cell each row is (plan: cell_id, line, gene, batch, pool), and a
    summary. With plan_only, no batch file is read and only "plan" and "report" are returned.

    `obs_sources` are LINE=PATH metadata tables (scripts/fetch_xatlas_obs.py output),
    `split_dir` holds genes.csv, controls.csv.gz and split.json, and `data` is a directory of
    the batch files or hf://datasets/Xaira-Therapeutics/X-Atlas-Orion/data.
    """
    split_dir = Path(split_dir)
    settings = json.loads((split_dir / "split.json").read_text())["settings"]
    if {source_line, target_line} - set(settings["lines"]):
        raise ValueError(f"the split covers {settings['lines']}, not {source_line} and {target_line}")
    control_label = settings["control_label"]
    genes = pd.read_csv(split_dir / "genes.csv", index_col="gene", keep_default_na=False)
    controls = pd.read_csv(split_dir / "controls.csv.gz", dtype=str)
    obs = read_metadata(obs_sources, gene_col="gene_target", batch_col="sample", cell_id_col="cell_barcode")

    plan, report = plan_cells(obs, genes, controls, source_line, target_line, control_label, subset)
    columns = vocabulary_columns(gene_metadata, vocab, aliases)
    report["vocabulary_genes_covered"] = int(len(set(columns[columns >= 0])))
    report["batch_files_to_read"] = int(plan["batch"].nunique())
    log("[cellline] plan: " + json.dumps({k: v for k, v in report.items() if not k.endswith("_genes")}))
    if plan_only:
        return {"plan": plan, "report": report}

    counts, library = load_counts(plan, data, columns, len(vocab), log=log)
    rows = np.arange(len(plan))
    split = Split(
        source_line=source_line,
        target_line=target_line,
        control_label=control_label,
        train_genes=report["train_genes"],
        val_genes=report["val_genes"],
        test_genes=sorted(genes.index[genes["split"] == "test"]),
        train_controls=rows[(plan["pool"] == "train").to_numpy()],
        val_controls=rows[(plan["pool"] == "val").to_numpy()],
    )
    cell_data = CellLineData(counts=counts, genes=list(vocab), line=plan["line"].to_numpy(dtype=str),
                             target=plan["gene"].to_numpy(dtype=str))
    check_split(cell_data, split)
    return {"data": cell_data, "split": split, "library": library, "plan": plan, "report": report}
