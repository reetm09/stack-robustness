"""Fetch only the per-cell metadata columns of X-Atlas/Orion from Hugging Face (no expression)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/Xaira-Therapeutics/X-Atlas-Orion"
COLS = ["cell_barcode", "sample", "num_features", "guide_target", "gene_target",
        "n_genes_by_counts", "total_counts", "pct_counts_mt", "pass_guide_filter"]
out = Path(sys.argv[1])
fs = HfFileSystem()
files = sorted(fs.glob(f"{REPO}/data/*.parquet"))
print(len(files), "files", flush=True)

def read(p):
    for attempt in range(5):
        try:
            with fs.open(p, block_size=8 * 2**20) as fh:
                return pq.ParquetFile(fh).read(columns=COLS).to_pandas()
        except Exception as e:
            err = e
    raise RuntimeError(f"{p}: {err}")

frames = {}
with ThreadPoolExecutor(16) as ex:
    for i, (p, df) in enumerate(zip(files, ex.map(read, files)), 1):
        frames.setdefault(Path(p).name.split("_Batch")[0], []).append(df)
        if i % 25 == 0: print(i, "done", flush=True)
for line, dfs in frames.items():
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{line}_obs.parquet", index=False)
    print(line, len(df), "cells,", df["sample"].nunique(), "batches", flush=True)
