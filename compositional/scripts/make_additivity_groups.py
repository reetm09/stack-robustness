"""Label every Norman double as additive, buffering/synergy or interaction, and save the labels.

In short: for each of the 131 measured pairs, this checks how well "A's effect + B's effect"
matches the real double, on a ruler from 0 (as good as repeating the experiment) to 1 (as bad as
guessing "nothing changes"). The 25% of pairs furthest from 0 are non-additive. Among those, the
pairs that stay far even when each gene's effect may be stretched or shrunk are "interaction";
the rest are "buffering/synergy". No model is involved; the labels come from the data only.

    cd stack-robustness
    python compositional/scripts/make_additivity_groups.py

Technical details:
- Rule: `stack_compositional.additivity.additivity_groups`, with q = 0.75 and one cut used for both
  scores.
- Scores: `additivity_scores`, on the `hvg2000` panel of the shared context. Truth and duplicate
  come from `splits/norman2019_halves.csv.gz`. wMSE weights come from the t-test of each first half
  vs all other perturbed first-half cells, as in the shared scoring.
- Output: `splits/compositional/additivity_groups.csv` (pair, plain, stretched, c1, c2, group,
  cut).
"""

import argparse
from pathlib import Path

import pandas as pd

from stack_compositional.additivity import additivity_groups, additivity_scores
from stack_compositional.contexts import load_norman
from stack_robustness.de import rank_genes
from stack_robustness.evaluation import EvalContext

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", default="hvg2000")
    parser.add_argument("--q", type=float, default=0.75)
    parser.add_argument("--context", type=Path, default=ROOT / "data/processed/compositional/s_shared_halves.npz")
    parser.add_argument("--halves", type=Path, default=ROOT / "splits/norman2019_halves.csv.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "splits/compositional/additivity_groups.csv")
    args = parser.parse_args()

    panel = EvalContext.load(args.context).panels[args.panel]
    adata = load_norman(ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad",
                        ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    adata = adata[:, panel].copy()
    halves = pd.read_csv(args.halves, index_col=0)["half"]
    first = (halves.reindex(adata.obs_names) == "first").to_numpy()
    perturbed = (adata.obs["condition"] != "control").to_numpy()
    de_rest = rank_genes(adata[first & perturbed].copy(), "condition", reference="rest")

    scores = additivity_scores(adata, halves, panel, de_rest)
    labels, cut = additivity_groups(scores["plain"], scores["stretched"], q=args.q)
    out = scores.assign(group=labels, cut=cut, panel=args.panel).rename_axis("pair").reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"cut = {cut:.4f}; groups: {labels.value_counts().to_dict()}; wrote {args.out}")


if __name__ == "__main__":
    main()
