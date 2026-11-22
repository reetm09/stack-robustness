"""Frozen STACK (in-context learning, no weight updates) on Norman held-out doubles.

For a held-out double A+B, STACK is prompted with cells where A is switched on and
queried with cells where B is switched on, and the reverse. Both use the "infer" half of
each single's cells, the same cells test-time training will predict from. Generated cells
become log-normalized profiles. They are scored against the same truth halves, baselines
and gene panels as scripts/norman_baselines.py, through its saved evaluation context.

Models reported, for each number of generation steps T:
  icl_T{T}_one  prompt A, query B; median over draws
  icl_T{T}      median over draws within each order, then over the two orders

    uv run python scripts/norman_icl.py --doubles 3 --steps 1 5 --draws 2   # from ttt/
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from stack_robustness.evaluation import EvalContext, markdown_table, median_tables
from stack_robustness.genes import load_stack_genes
from stack_ttt.model import load_stack
from stack_ttt.norman import load_norman, single_pools
from stack_ttt.predict import generate, mean_log_profile, median_of_medians

ROOT = Path(__file__).resolve().parents[2]
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/raw/stack/bc_large_aligned.ckpt")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--context", type=Path, default=ROOT / "data/processed/norman2019_eval.npz")
    parser.add_argument("--splits", type=Path, default=ROOT / "splits/norman2019_conditions.csv")
    parser.add_argument("--doubles", nargs="+", default=["all"], help="held-out doubles, a count, or 'all'")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 5], help="generation steps T")
    parser.add_argument("--draws", type=int, default=3, help="generation draws per prompt order")
    parser.add_argument("--seed", type=int, default=0, help="seed for the adapt/infer split of single cells")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ttt/norman_icl")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = EvalContext.load(args.context)
    split = pd.read_csv(args.splits)
    held_out = sorted(split.loc[split["split"].eq("test") & split["kind"].eq("combination"), "condition"])
    if args.doubles == ["all"]:
        doubles = held_out
    elif len(args.doubles) == 1 and args.doubles[0].isdigit():
        doubles = held_out[: int(args.doubles[0])]
    else:
        doubles = args.doubles
    assert set(doubles) <= set(held_out), "only held-out doubles can be scored"
    singles = sorted({g for d in doubles for g in d.split("+")})
    log(f"{len(doubles)} held-out doubles: {doubles}")

    vocab = load_stack_genes(args.vocab)
    cells = load_norman(args.raw, vocab, conditions=singles)
    pools = single_pools(cells.condition, singles, seed=args.seed)
    scored_genes = list(ctx.truth.columns)
    log(f"loaded {cells.counts.shape[0]} single-perturbation cells on {len(vocab)} STACK genes")

    model, hparams = load_stack(args.checkpoint, device=device)
    log(f"STACK on {device}: model_config={hparams['model_config']}, n_kept_cell={hparams.get('n_kept_cell')}")

    predictions, timing = {}, {}
    for steps in args.steps:
        one, both = {}, {}
        tic = time.time()
        for double in doubles:
            a, b = double.split("+")
            by_order = {}
            for prompt, query in [(a, b), (b, a)]:
                p_rows, q_rows = pools[(prompt, "infer")], pools[(query, "infer")]
                by_order[f"{prompt}>{query}"] = [
                    mean_log_profile(
                        generate(model, cells.counts[p_rows], cells.counts[q_rows], vocab, args.vocab,
                                 num_steps=steps, seed=1000 * draw + 1, batch_size=args.batch_size),
                        cells.library[q_rows],
                        vocab,
                        scored_genes,
                    )
                    for draw in range(args.draws)
                ]
            one[double] = pd.concat(by_order[f"{a}>{b}"], axis=1).median(axis=1)
            both[double] = median_of_medians(by_order)
            log(f"T={steps} {double} done")
        timing[f"T{steps}_seconds_per_double"] = (time.time() - tic) / len(doubles)
        predictions[f"icl_T{steps}_one"] = pd.DataFrame(one).T
        predictions[f"icl_T{steps}"] = pd.DataFrame(both).T
    if device == "cuda":
        timing["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    long = ctx.score(predictions, perturbations=doubles)
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out / "scores.csv", index=False)
    values, gaps = median_tables(long, "value"), median_tables(long, "gap_closed")
    summary = [
        "# Frozen STACK (ICL) on Norman held-out doubles\n",
        f"{len(doubles)} doubles: {', '.join(doubles)}. {args.draws} draws per prompt order. "
        f"Timing: {json.dumps({k: round(v, 2) for k, v in timing.items()})}\n",
    ]
    for panel in values:
        summary += [
            f"## Panel `{panel}`\n",
            "Median score:\n",
            markdown_table(values[panel]) + "\n",
            "Median share of the gap from each metric's negative control to a perfect score:\n",
            markdown_table(gaps[panel]) + "\n",
        ]
    (args.out / "summary.md").write_text("\n".join(summary))
    log(f"wrote {args.out}/summary.md; timing {timing}")


if __name__ == "__main__":
    main()
