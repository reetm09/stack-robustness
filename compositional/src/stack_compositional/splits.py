"""Experiment 2 splits: S-all folds and S-gene held-out-gene splits (design in SPLITS.md).

In short:
- **S-all** grades every pair with frozen STACK. Only the `weighted` arm's two weights are fitted,
  so the 131 pairs are dealt into 5 folds, and each fold's weights come from the other 4.
- **S-gene** hides about 11 genes. Every pair containing one of them is a test pair. The model
  never sees those genes in a training pair (it still sees their single-gene measurements). Test
  pairs are labelled by how many of their two genes appear in training pairs: 2/2, 1/2 or 0/2.

Technical details:
- Pairs are unordered condition names "A+B". Both prompt orders of a pair always share its fold or
  split, so an order cannot leak through its mirror.
- `category` is computed from the final training doubles, not from the draw. A "random" pair whose
  gene only occurs in that one pair is therefore labelled 1/2, not 2/2.
- `s_gene_split` draws from one `numpy.random.default_rng(seed)` until the checks pass and records
  the number of tries. The result depends only on the seed and the sorted inputs.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORIES = ("seen 2/2", "unseen 1/2", "unseen 2/2")


def genes_of(pair: str) -> tuple[str, str]:
    a, b = pair.split("+")
    return a, b


def all_genes(pairs) -> list[str]:
    return sorted({g for p in pairs for g in genes_of(p)})


def category(pair: str, train_pairs) -> str:
    """How many of the pair's genes appear in any training double: 2 → seen 2/2, 1 → unseen 1/2, 0 → unseen 2/2."""
    seen = set(all_genes(train_pairs))
    n = sum(g in seen for g in genes_of(pair))
    return {2: "seen 2/2", 1: "unseen 1/2", 0: "unseen 2/2"}[n]


def s_all_folds(pairs, groups: pd.Series | None = None, k: int = 5, seed: int = 0) -> pd.Series:
    """Deal the pairs into k folds, stratified by additivity group so each fold has non-additive pairs."""
    pairs = sorted(pairs)
    rng = np.random.default_rng(seed)
    strata = (groups.reindex(pairs).fillna("?") if groups is not None
              else pd.Series("all", index=pairs))
    fold = pd.Series(-1, index=pairs, name="fold")
    offset = 0
    for name in sorted(strata.unique()):
        members = list(rng.permutation(strata.index[strata == name].to_numpy()))
        for i, p in enumerate(members):
            fold[p] = (offset + i) % k
        offset += len(members)  # continue the round-robin so fold sizes stay within 1
    return fold


@dataclass
class HoldOut:
    """One held-out-gene draw: the held genes, the held pairs with categories, and the tries used."""

    genes: list[str]
    held: pd.Series  # pair -> category
    train: list[str]
    tries: int
    counts: dict = field(default_factory=dict)


def hold_out_genes(pairs, seed: int, frac_genes: float = 0.15, frac_seen: float = 0.15,
                   min_per_category: int = 5, min_train_frac: float = 0.6, max_tries: int = 1000) -> HoldOut:
    """Held-out-gene split of `pairs` (S-gene step 1–3 and 5).

    Draw a gene set G (round(frac_genes · #genes)). Held = every pair with a gene in G, plus a random
    frac_seen of the remaining pairs. Train = the rest. Accept the first draw where every category has
    at least `min_per_category` held pairs and at least `min_train_frac` of pairs are left for
    training.
    """
    pairs = sorted(pairs)
    genes = all_genes(pairs)
    n_genes = max(1, round(frac_genes * len(genes)))
    rng = np.random.default_rng(seed)
    for tries in range(1, max_tries + 1):
        held_genes = sorted(rng.choice(genes, size=n_genes, replace=False).tolist())
        touched = [p for p in pairs if set(genes_of(p)) & set(held_genes)]
        rest = [p for p in pairs if p not in touched]
        n_seen = round(frac_seen * len(rest))
        random_held = sorted(rng.choice(rest, size=n_seen, replace=False).tolist()) if n_seen else []
        held = sorted(touched + random_held)
        train = [p for p in rest if p not in random_held]
        cats = pd.Series({p: category(p, train) for p in held}, name="category")
        counts = {c: int((cats == c).sum()) for c in CATEGORIES}
        if min(counts.values()) >= min_per_category and len(train) >= min_train_frac * len(pairs):
            return HoldOut(held_genes, cats, train, tries, counts)
    raise RuntimeError(f"no draw passed the checks in {max_tries} tries (seed {seed})")


def s_gene_split(doubles, singles, seed: int, test_kw: dict | None = None, val_kw: dict | None = None):
    """One S-gene seed: the test split and, inside its training doubles, the validation split.

    Returns (test_table, val_table, meta). Tables follow the team format (condition, kind, split)
    plus `category`:
    - test_table: singles + train doubles (including the validation doubles) = "train"; held = "test";
    - val_table: singles + inner train doubles = "train"; validation doubles = "val"; test doubles
      are left out entirely, as in `splits/norman2019_val_conditions.csv`.
    """
    test = hold_out_genes(doubles, seed, **(test_kw or {}))
    val = hold_out_genes(test.train, seed + 10_000, **(val_kw or {}))

    def table(held: HoldOut, held_name: str) -> pd.DataFrame:
        rows = [(s, "single", "train", "") for s in sorted(singles)]
        rows += [(p, "combination", "train", "") for p in held.train]
        rows += [(p, "combination", held_name, held.held[p]) for p in held.held.index]
        out = pd.DataFrame(rows, columns=["condition", "kind", "split", "category"])
        return out.sort_values("condition", ignore_index=True)

    test_table = table(HoldOut(test.genes, test.held, sorted(test.train), test.tries), "test")
    def info(h: HoldOut, pool, s: int) -> dict:
        # genes not drawn but still unseen: their only pairs were among the random held pairs
        unseen = sorted(set(all_genes(pool)) - set(all_genes(h.train)) - set(h.genes))
        return {"seed": s, "held_genes": h.genes, "also_unseen": unseen, "tries": h.tries, "counts": h.counts,
                "n_train_doubles": len(h.train)}

    meta = {"test": info(test, doubles, seed), "val": info(val, test.train, seed + 10_000)}
    return test_table, table(val, "val"), meta
