"""Write the Experiment 2 split files: S-all folds and S-gene seeds 0–4 (design in SPLITS.md).

In short: this decides, once and for all, which gene pairs are hidden in each experiment, and saves
those choices so every run uses the same ones. No model or expression data is involved; only the
list of pairs (and, for balancing S-all folds, the additivity groups).

    cd stack-robustness
    python compositional/scripts/make_splits.py

Technical details:
- Pairs and singles come from `splits/norman2019_conditions.csv` (all 131 doubles, 105 singles).
- `splits/compositional/s_all_folds.csv` (condition, fold, group): 5 folds, seed 0, stratified by
  additivity group.
- For each seed k in 0..4 (team format: condition, kind, split, plus category):
  - `s_gene_seed{k}_conditions.csv`: split = train / test (train includes the validation doubles);
  - `s_gene_seed{k}_val_conditions.csv`: split = train / val; test doubles are left out.
  Use either with `build_contexts.py --conditions ... --halves splits/norman2019_halves.csv.gz
  --held test|val`. The halves file is per cell, so it is the same for every split.
- `s_gene_meta.json`: held-out genes, tries and category counts per seed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from stack_compositional.splits import s_all_folds, s_gene_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--conditions", type=Path, default=ROOT / "splits/norman2019_conditions.csv")
    parser.add_argument("--groups", type=Path, default=ROOT / "splits/compositional/additivity_groups.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "splits/compositional")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    cond = pd.read_csv(args.conditions)
    doubles = sorted(cond.loc[cond["kind"] == "combination", "condition"])
    singles = sorted(cond.loc[cond["kind"] == "single", "condition"])
    groups = pd.read_csv(args.groups, index_col="pair")["group"]
    args.out.mkdir(parents=True, exist_ok=True)

    folds = s_all_folds(doubles, groups, k=5, seed=0)
    folds.rename_axis("condition").to_frame().assign(group=groups.reindex(folds.index)).reset_index() \
        .to_csv(args.out / "s_all_folds.csv", index=False)
    print(f"s_all_folds.csv: fold sizes {folds.value_counts().sort_index().tolist()}")

    meta = {}
    for seed in args.seeds:
        test, val, m = s_gene_split(doubles, singles, seed)
        test.to_csv(args.out / f"s_gene_seed{seed}_conditions.csv", index=False)
        val.to_csv(args.out / f"s_gene_seed{seed}_val_conditions.csv", index=False)
        meta[f"seed{seed}"] = m
        print(f"seed {seed}: test {m['test']['counts']} (train {m['test']['n_train_doubles']}), "
              f"val {m['val']['counts']} (train {m['val']['n_train_doubles']})")
    (args.out / "s_gene_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
