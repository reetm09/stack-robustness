"""Score the shared baselines on Norman et al. (2019) double perturbations (Study 2 setup).

Every single perturbation and 80% of the doubles count as training data; the other 20%
of doubles are held out and scored. Each held-out double's cells are split in half:
the first half is the ground truth every model is scored against, the second half is
the technical duplicate (Miller et al.).

Deliberate differences from Miller et al.'s Norman pipeline: every cell is kept (they
cap each perturbation at the mean cell count and control at 8,192 cells), and the genes
are STACK's vocabulary rather than 8,192 HVGs plus the perturbed genes, so 17 of the 105
perturbed genes are not scored.

    uv run python scripts/norman_baselines.py

Inputs:  data/raw/norman/NormanWeissman2019_filtered.h5ad (scPerturb, Zenodo 13350497)
         data/raw/stack_vocab/basecount_1000per_15000max.pkl (Stack-Large-Aligned on Hugging Face)
Outputs: splits/norman2019_conditions.csv  train/test assignment of every condition
         splits/norman2019_halves.csv.gz   truth ("first") / duplicate ("second") half of every cell
         data/processed/norman2019_eval.npz evaluation context for scoring any model (EvalContext)
         results/norman_baselines/scores.csv, summary.md
"""

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from stack_robustness.baselines import additive, assign_halves, interpolated_duplicate, mean_baseline, pseudobulk
from stack_robustness.data import log_normalize, split_conditions
from stack_robustness.de import rank_genes
from stack_robustness.evaluation import EvalContext, markdown_table, median_tables
from stack_robustness.genes import load_stack_genes

RAW = Path("data/raw/norman/NormanWeissman2019_filtered.h5ad")
VOCAB = Path("data/raw/stack_vocab/basecount_1000per_15000max.pkl")
CONTROL = "control"
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def load_log_normalized(min_cells: int) -> ad.AnnData:
    """Log-normalized expression of STACK's genes detected in >= min_cells cells.

    Library sizes are each cell's total counts over all genes (scPerturb's `ncounts`).
    """
    backed = ad.read_h5ad(RAW, backed="r")
    in_vocab = backed.var_names.isin(set(load_stack_genes(VOCAB)))
    X = sparse.csc_matrix(backed.X[:, in_vocab])  # stored column-major: reads only these genes
    obs = backed.obs[["perturbation", "gemgroup"]].copy()
    library = backed.obs["ncounts"].to_numpy()
    genes = backed.var_names[in_vocab]
    backed.file.close()
    assert (np.asarray(X.sum(axis=1)).ravel() <= library + 1e-3).all(), "ncounts is not total counts"
    detected = np.diff(X.indptr) >= min_cells  # stored entries per gene = cells with nonzero counts
    X = X[:, detected]
    X = log_normalize(X.tocsr(), library)
    obs["condition"] = obs["perturbation"].astype(str).str.replace("_", "+", regex=False)
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes[detected]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--min-cells", type=int, default=3, help="drop genes detected in fewer cells")
    parser.add_argument("--out", type=Path, default=Path("results/norman_baselines"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--context", type=Path, default=Path("data/processed/norman2019_eval.npz"))
    args = parser.parse_args()

    adata = load_log_normalized(args.min_cells)
    genes = list(adata.var_names)
    cond = adata.obs["condition"]
    log(f"loaded {adata.n_obs} cells x {adata.n_vars} STACK genes, {cond.nunique()} conditions")

    split = split_conditions(cond, CONTROL, args.seed, args.test_fraction)
    test = split.loc[split["split"] == "test", "condition"].tolist()
    train = split.loc[split["split"] == "train", "condition"].tolist()
    halves = assign_halves(cond, seed=args.seed)
    first = (halves == "first").to_numpy()
    second = (halves == "second").to_numpy()
    is_ctrl = (cond == CONTROL).to_numpy()
    args.splits.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.splits / "norman2019_conditions.csv", index=False)
    halves.rename("half").rename_axis("cell").to_csv(args.splits / "norman2019_halves.csv.gz")
    log(f"{len(test)} test doubles, {len(train)} training conditions")

    profiles = pseudobulk(adata.X, cond, genes)
    truth = pseudobulk(adata.X[first], cond[first], genes).loc[test]
    duplicate = pseudobulk(adata.X[second], cond[second], genes).loc[test]
    ctrl = profiles.loc[CONTROL]
    mean = mean_baseline(profiles, train)

    # As in Miller et al.: ground-truth DE on the truth halves of perturbed cells only;
    # the interpolation's DE on the duplicate halves of every condition, control included.
    log("DE of each perturbation's truth half against all other perturbed cells (t-test)")
    de_rest_truth = rank_genes(adata[first & ~is_ctrl].copy(), "condition", reference="rest")
    log("DE of each condition's duplicate half against all other cells (t-test)")
    de_rest_dup = rank_genes(adata[second].copy(), "condition", reference="rest")
    log("DE of each test double's truth half against control cells (Wilcoxon)")
    vs_ctrl = (cond.isin(test).to_numpy() & first) | is_ctrl
    de_ctrl_truth = rank_genes(adata[vs_ctrl].copy(), "condition", reference=CONTROL, method="wilcoxon")

    preds = {
        "control_baseline": pd.DataFrame([ctrl] * len(test), index=test),
        "mean_baseline": pd.DataFrame([mean] * len(test), index=test),
        "additive": pd.DataFrame({t: additive(ctrl, profiles, t.split("+")) for t in test}).T,
        "technical_duplicate": duplicate,
        "interpolated_duplicate": pd.DataFrame(
            {t: interpolated_duplicate(duplicate.loc[t], mean, de_rest_dup[t]["pval_adj"]) for t in test}
        ).T,
    }

    train_and_ctrl = cond.isin(train).to_numpy() | is_ctrl
    hv = sc.pp.highly_variable_genes(adata[train_and_ctrl].copy(), n_top_genes=args.n_hvg, inplace=False)
    panels = {"stack_genes": genes, f"hvg{args.n_hvg}": list(adata.var_names[hv["highly_variable"].to_numpy()])}

    ctx = EvalContext(
        truth=truth,
        ctrl=ctrl,
        pert_ref=mean,
        de_vs_ctrl={t: de_ctrl_truth[t] for t in test},
        de_vs_rest={t: de_rest_truth[t] for t in test},
        baselines=preds,
        panels=panels,
    )
    args.context.parent.mkdir(parents=True, exist_ok=True)
    ctx.save(args.context)
    log(f"saved the evaluation context to {args.context}")

    log("scoring the baselines")
    long = ctx.score({})
    n_train_doubles = int((split["kind"].eq("combination") & split["split"].eq("train")).sum())
    summary = [
        f"# Norman et al. (2019) baselines, Study 2 setup (seed {args.seed})\n",
        f"{len(test)} held-out doubles scored against their truth half; {len(train)} training conditions "
        f"(all singles plus {n_train_doubles} doubles); {int(is_ctrl.sum())} control cells.\n",
    ]
    values, gaps = median_tables(long, "value"), median_tables(long, "gap_closed")
    for panel, cols in panels.items():
        summary += [
            f"## Panel `{panel}` ({len(cols)} genes)\n",
            "Median score over held-out doubles:\n",
            markdown_table(values[panel]) + "\n",
            "Median share of the gap from each metric's negative control to a perfect score "
            "(the `interpolated_duplicate` column is Miller et al.'s dynamic range fraction):\n",
            markdown_table(gaps[panel]) + "\n",
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out / "scores.csv", index=False)
    (args.out / "summary.md").write_text("\n".join(summary))
    log(f"wrote {args.out}/scores.csv and summary.md")


if __name__ == "__main__":
    main()
