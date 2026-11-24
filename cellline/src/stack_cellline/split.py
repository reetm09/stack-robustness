"""The Study 1 split for X-Atlas/Orion, by the proposal's rule (PDF p. 3; proposal/main.tex:106).

- Eligible targets have at least `min_cells` cells in every line given, counted only in GEM
  batches that hold that line's non-targeting controls. Pass only cells that pass the guide
  filter, or filter with --query: what "guide-filtered" means is still a team question.
- Eligible targets are split into train, validation and test by gene identity, so every guide
  of a gene lands in the same split. One gene split serves both transfer directions, so a
  test gene can never be a training gene in the other direction.
- Each line's non-targeting control cells are split into train, validation and test pools
  within each GEM batch.

Settings chosen on this branch, not yet agreed by the whole team: 70/15/15 for genes and for
controls, seed 0, one gene split shared by both directions.

Run once on the real cell metadata (from an .h5ad only `obs` is read; no expression data):

    PYTHONPATH=cellline/src python -m stack_cellline.split \\
        --obs <file.h5ad> --line-col <col> --gene-col <col> --batch-col <col> \\
        --control-label <label> --lines HEK293T HCT116 --out cellline/splits/xatlas_orion

Outputs in --out, identical byte for byte on every rerun with the same inputs and settings:
  genes.csv         one row per target gene: split (train/val/test/excluded), cells per line
                    in batches with controls, guides, and why an excluded gene failed
  controls.csv.gz   one row per pooled control cell: cell_id, line, batch, pool
  split.json        the settings and counts
"""

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("train", "val", "test")
EXCLUDED = "excluded"
OUTPUTS = ("genes.csv", "controls.csv.gz", "split.json")


def _rng(seed: int, *names) -> np.random.Generator:
    """A generator that depends only on the seed and the names, not on row order."""
    return np.random.default_rng(np.random.SeedSequence([seed, *(zlib.crc32(str(n).encode()) for n in names)]))


def _sizes(n: int, fractions) -> tuple:
    """Counts per stage: validation and test rounded, training takes the rest."""
    n_val, n_test = int(round(fractions[1] * n)), int(round(fractions[2] * n))
    if n_val + n_test > n:
        raise ValueError(f"fractions {fractions} leave no training items out of {n}")
    return n - n_val - n_test, n_val, n_test


def _control_batches(cells: pd.DataFrame, line: str, control_label: str, min_batch_controls: int) -> list:
    ctrl = cells[(cells["line"] == line) & (cells["gene"] == control_label)]
    sizes = ctrl.groupby("batch").size()
    return sorted(sizes[sizes >= min_batch_controls].index)


def eligible_genes(cells, lines, control_label, min_cells=100, min_batch_controls=1) -> pd.DataFrame:
    """One row per target gene: its cells per line in batches with that line's controls, its
    number of guides (if a guide column is present), whether it is eligible, and why not."""
    targets = cells[cells["gene"] != control_label]
    genes = sorted(set(targets["gene"]))
    table = pd.DataFrame(index=pd.Index(genes, name="gene"))
    for line in lines:
        batches = _control_batches(cells, line, control_label, min_batch_controls)
        kept = targets[(targets["line"] == line) & targets["batch"].isin(batches)]
        table[f"n_cells_{line}"] = kept["gene"].value_counts().reindex(genes, fill_value=0).astype(int)
    if "guide" in cells:
        table["n_guides"] = targets.groupby("gene")["guide"].nunique().reindex(genes).astype(int)
    reasons = []
    for gene in genes:
        short = [f"{line}: {table.at[gene, f'n_cells_{line}']} < {min_cells}" for line in lines
                 if table.at[gene, f"n_cells_{line}"] < min_cells]
        reasons.append("; ".join(short))
    table["reason"] = reasons
    table["eligible"] = table["reason"] == ""
    return table


def split_genes(genes, fractions, seed: int) -> dict:
    """gene -> train/val/test, by a seeded shuffle of the sorted genes."""
    genes = sorted(genes)
    order = _rng(seed, "genes").permutation(len(genes))
    stages = np.repeat(STAGES, _sizes(len(genes), fractions))
    return {genes[i]: str(stage) for i, stage in zip(order, stages)}


def control_pools(cells, lines, control_label, fractions, seed: int, min_batch_controls=1) -> pd.DataFrame:
    """Each line's control cells, split into train/val/test pools within each GEM batch."""
    frames = []
    for line in lines:
        for batch in _control_batches(cells, line, control_label, min_batch_controls):
            mask = (cells["line"] == line) & (cells["batch"] == batch) & (cells["gene"] == control_label)
            ids = np.sort(cells.loc[mask, "cell_id"].to_numpy(dtype=object))
            order = _rng(seed, "controls", line, batch).permutation(len(ids))
            pools = np.repeat(STAGES, _sizes(len(ids), fractions))
            frames.append(pd.DataFrame({"cell_id": ids[order], "line": line, "batch": batch, "pool": pools}))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["line", "batch", "cell_id"], kind="mergesort").reset_index(drop=True)


def build_split(cells, lines, control_label, min_cells=100, fractions=(0.7, 0.15, 0.15), seed=0,
                min_batch_controls=1):
    """(genes table, control pools, summary) from per-cell metadata with columns cell_id, line,
    gene, batch and optionally guide."""
    missing = {"cell_id", "line", "gene", "batch"} - set(cells.columns)
    if missing:
        raise ValueError(f"cell table lacks columns {sorted(missing)}")
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions}")
    lines = [str(line) for line in lines]
    if len(set(lines)) < 2:
        raise ValueError("give at least two distinct cell lines")
    n_read = len(cells)
    cells = cells[cells["line"].isin(lines)]
    if cells["cell_id"].duplicated().any():
        dup = cells.loc[cells["cell_id"].duplicated(), "cell_id"].iloc[0]
        raise ValueError(f"cell ids must be unique; {dup!r} repeats")
    for line in lines:
        if not (cells["line"] == line).any():
            raise ValueError(f"no cells of line {line!r}")
        if not _control_batches(cells, line, control_label, min_batch_controls):
            raise ValueError(f"line {line!r} has no GEM batch with {control_label!r} controls")

    genes = eligible_genes(cells, lines, control_label, min_cells, min_batch_controls)
    eligible = list(genes.index[genes["eligible"]])
    if len(eligible) < 3:
        raise ValueError(f"only {len(eligible)} eligible genes; need at least one per stage")
    stage = split_genes(eligible, fractions, seed)
    genes.insert(0, "split", [stage.get(g, EXCLUDED) for g in genes.index])
    if any((genes["split"] == s).sum() == 0 for s in STAGES):
        raise ValueError("a stage got no genes; use more eligible genes or other fractions")
    genes = genes.drop(columns="eligible")
    controls = control_pools(cells, lines, control_label, fractions, seed, min_batch_controls)

    summary = {
        "settings": {
            "lines": lines,
            "control_label": control_label,
            "min_cells": min_cells,
            "min_batch_controls": min_batch_controls,
            "fractions": dict(zip(STAGES, fractions)),
            "seed": seed,
        },
        "cells": {"read": int(n_read), "in_lines": int(len(cells))},
        "genes": {"targets": int(len(genes)), **{s: int((genes["split"] == s).sum()) for s in (*STAGES, EXCLUDED)}},
        "controls": {
            line: {
                "batches_with_controls": len(_control_batches(cells, line, control_label, min_batch_controls)),
                "batches_without_controls": int(cells.loc[cells["line"] == line, "batch"].nunique())
                - len(_control_batches(cells, line, control_label, min_batch_controls)),
                **{s: int(((controls["line"] == line) & (controls["pool"] == s)).sum()) for s in STAGES},
            }
            for line in lines
        },
    }
    return genes, controls, summary


def read_cells(sources, gene_col, batch_col, line_col=None, cell_id_col=None, guide_col=None, query=None):
    """Per-cell metadata from .h5ad (obs only), .csv/.tsv (optionally .gz) or .parquet files.

    With `line_col`, each source is a path. Without it, each source is LINE=PATH, for files
    that hold one cell line each. Cell ids come from `cell_id_col`, or from an .h5ad's obs
    names. `query` (pandas DataFrame.query syntax) filters cells before anything is counted.
    """
    frames = []
    for source in sources:
        line, path = (None, source) if line_col else _line_and_path(source)
        path = Path(path)
        frame, index_ids = _read_table(path)
        if query:
            frame = frame.query(query)
        wanted = [c for c in (gene_col, batch_col, line_col, cell_id_col, guide_col) if c]
        absent = [c for c in wanted if c not in frame.columns]
        if absent:
            raise ValueError(f"{path} has no column(s) {absent}; it has {list(frame.columns)}")
        if cell_id_col:
            ids = frame[cell_id_col]
        elif index_ids:
            ids = frame.index.to_series()
        else:
            raise ValueError(f"{path} is a table: name its cell id column with --cell-id-col")
        out = pd.DataFrame({
            "cell_id": ids.to_numpy(),
            "line": frame[line_col].to_numpy() if line_col else line,
            "gene": frame[gene_col].to_numpy(),
            "batch": frame[batch_col].to_numpy(),
        })
        if guide_col:
            out["guide"] = frame[guide_col].to_numpy()
        for column in ("cell_id", "line", "gene", "batch"):
            if out[column].isna().any():
                raise ValueError(f"{path}: {int(out[column].isna().sum())} cells have no {column}")
            out[column] = out[column].astype(str)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _line_and_path(source: str):
    line, sep, path = source.partition("=")
    if not sep or not line or not path:
        raise ValueError(f"without --line-col, give each file as LINE=PATH, not {source!r}")
    return line, path


def _read_table(path: Path):
    """(frame, whether its index holds cell ids)."""
    name = path.name.lower()
    if name.endswith(".h5ad"):
        import anndata as ad

        adata = ad.read_h5ad(path, backed="r")
        try:
            return adata.obs.copy(), True
        finally:
            adata.file.close()
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path), False
    if name.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t"), False
    if name.endswith(".parquet"):
        return pd.read_parquet(path), False
    raise ValueError(f"unsupported file type: {path}")


def write_split(out, genes, controls, summary, overwrite=False) -> None:
    out = Path(out)
    existing = [name for name in OUTPUTS if (out / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out} already holds {existing}; pass --overwrite to replace a split")
    out.mkdir(parents=True, exist_ok=True)
    genes.to_csv(out / "genes.csv")
    controls.to_csv(out / "controls.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (out / "split.json").write_text(json.dumps(summary, indent=1) + "\n")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--obs", nargs="+", required=True,
                        help="cell metadata files (.h5ad, .csv, .tsv, .parquet); LINE=PATH without --line-col")
    parser.add_argument("--gene-col", required=True, help="column naming each cell's target gene or control")
    parser.add_argument("--batch-col", required=True, help="column naming each cell's GEM batch")
    parser.add_argument("--control-label", required=True, help="value of --gene-col for non-targeting controls")
    parser.add_argument("--lines", nargs="+", required=True, help="cell lines, as named in the data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--line-col", help="column naming each cell's line (else give LINE=PATH files)")
    parser.add_argument("--cell-id-col", help="column of unique cell ids (default: an .h5ad's obs names)")
    parser.add_argument("--guide-col", help="column naming each cell's guide; only counted, for the record")
    parser.add_argument("--query", help="pandas query selecting the cells to use, e.g. a guide filter")
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--min-batch-controls", type=int, default=1,
                        help="controls a GEM batch needs to count as a batch with controls")
    parser.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing split in --out")
    args = parser.parse_args(argv)

    cells = read_cells(args.obs, args.gene_col, args.batch_col, args.line_col, args.cell_id_col,
                       args.guide_col, args.query)
    genes, controls, summary = build_split(cells, args.lines, args.control_label, args.min_cells,
                                           tuple(args.fractions), args.seed, args.min_batch_controls)
    summary["inputs"] = {
        "obs": list(args.obs),
        "columns": {k: getattr(args, k) for k in ("line_col", "gene_col", "batch_col", "cell_id_col", "guide_col")},
        "query": args.query,
    }
    write_split(args.out, genes, controls, summary, overwrite=args.overwrite)
    g = summary["genes"]
    print(f"{g['targets']} targets: {g['train']} train, {g['val']} val, {g['test']} test, {g['excluded']} excluded")
    for line, c in summary["controls"].items():
        print(f"{line} controls in {c['batches_with_controls']} batches: "
              f"{c['train']} train, {c['val']} val, {c['test']} test")
    print(f"wrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
