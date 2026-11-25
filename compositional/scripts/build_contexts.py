"""Build both scoring contexts (halves and all cells) for one split, and score the CPU baselines.

In short: for a given split this prepares the "right answers" for the held-out pairs in both
scoring modes (see `stack_compositional.contexts`). It then grades the simple baselines
(control, mean, additive, and in halves mode the duplicates), so the results table has its first
rows.

    cd stack-robustness            # repo root
    python compositional/scripts/build_contexts.py --split s_shared
    python compositional/scripts/build_contexts.py --split s_shared_val

Technical details:
- Splits:
  - `s_shared` = `splits/norman2019_conditions.csv` + `norman2019_halves.csv.gz`, held-out = "test";
  - `s_shared_val` = the `_val_` files, held-out = "val";
  - `--conditions` / `--halves` / `--held` point it at any other split file (S-all and S-gene, once
    step 2.7 writes them).
- Outputs:
  - `data/processed/compositional/<split>_{halves,all}.npz` (EvalContext);
  - `results/compositional/<split>/scores.csv` (the EvalContext long format plus a `truth` column);
  - `args.json`.
- Peak RAM is about 7 GB (DE on every perturbed cell), and a run takes 10–30 min on CPU.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from stack_compositional.contexts import TRUTH_MODES, build_context, load_norman

ROOT = Path(__file__).resolve().parents[2]  # stack-robustness/
SHARED = {
    "s_shared": ("splits/norman2019_conditions.csv", "splits/norman2019_halves.csv.gz", "test"),
    "s_shared_val": ("splits/norman2019_val_conditions.csv", "splits/norman2019_val_halves.csv.gz", "val"),
}
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", required=True, help="name; s_shared / s_shared_val have default files")
    parser.add_argument("--conditions", type=Path, default=None, help="CSV with condition, kind, split")
    parser.add_argument("--halves", type=Path, default=None, help="CSV(.gz) with cell, half")
    parser.add_argument("--held", default=None, help="value of the split column that is held out")
    parser.add_argument("--truth", nargs="+", default=list(TRUTH_MODES), choices=TRUTH_MODES)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    parser.add_argument("--n-hvg", type=int, default=2000)
    args = parser.parse_args()

    conditions, halves_path, held = SHARED.get(args.split, (None, None, None))
    args.conditions = args.conditions or ROOT / conditions
    args.halves = args.halves or ROOT / halves_path
    args.held = args.held or held

    split = pd.read_csv(args.conditions)
    test = split.loc[split["split"] == args.held, "condition"].tolist()
    train = split.loc[split["split"] == "train", "condition"].tolist()
    halves = pd.read_csv(args.halves, index_col=0)["half"]

    adata = load_norman(args.raw, args.vocab)
    keep = adata.obs["condition"].isin(set(test) | set(train) | {"control"}).to_numpy()
    adata = adata[keep].copy()  # e.g. the validation split never loads the test doubles
    log(f"{adata.n_obs} cells x {adata.n_vars} genes; {len(test)} held-out, {len(train)} training conditions")

    ctx_dir = ROOT / "data/processed/compositional"
    out_dir = ROOT / "results/compositional" / args.split
    ctx_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for mode in args.truth:
        log(f"building the {mode!r} context")
        ctx = build_context(adata, test, train, halves, truth=mode, n_hvg=args.n_hvg)
        ctx.save(ctx_dir / f"{args.split}_{mode}.npz")
        log(f"scoring baselines ({mode})")
        frames.append(ctx.score({}).assign(truth=mode))
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "scores.csv", index=False)
    (out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"wrote {out_dir / 'scores.csv'} and contexts in {ctx_dir}")


if __name__ == "__main__":
    main()
