"""TTT vs ICL for STACK on Norman held-out doubles (Study 4, combinatorial arm).

For each held-out double A+B:
  icl        frozen STACK prompted with A's cells and queried with B's cells, and the reverse
  ttt        the same prediction after fresh low-rank adapters are trained on A+B's tasks:
             training doubles sharing a gene with A+B (both prompt orders), the null control
             task if --null, and masked reconstruction of A's and B's own prompt cells
  nulladapt  (with --null-adapter) one adapter trained on the null control task alone, a
             control for TTT gains that come from fitting STACK to this dataset, not to A+B
  query      the query cells unchanged, no model: the floor a prediction must beat
Single-perturbation cells are split into "adapt" and "infer" halves, and predictions use
"infer" only. Everything is scored through the evaluation context saved by
scripts/norman_baselines.py, beside the same baselines and on the same genes.

Each model is reported raw (the generated profile) and diff (observed query plus generated
with prompt minus generated with a control prompt; Cell-Eval's generated-control reference,
which cancels shifts STACK applies whatever the prompt). Suffix _one: prompt A, query B only.

    uv run python scripts/norman_compare.py --doubles 3 --ttt-steps 50   # from ttt/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stack_robustness.evaluation import EvalContext, markdown_table, median_tables
from stack_robustness.genes import load_stack_genes
from stack_ttt.adapt import AdaptConfig, adapt
from stack_ttt.lora import adapters_disabled, inject_lora, stack_targets
from stack_ttt.model import load_stack
from stack_ttt.norman import load_norman, single_pools
from stack_ttt.objective import LossWeights
from stack_ttt.predict import predict_double
from stack_ttt.tasks import CONTROL, Task, control_pools, neighbours, norman_tasks

ROOT = Path(__file__).resolve().parents[2]
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def parse_weights(text: str) -> LossWeights:
    """"ed=1,recon=0,sw=0,cls=0" -> LossWeights; unnamed components keep their defaults."""
    values = {k: float(v) for k, v in (item.split("=") for item in text.split(",") if item)}
    return LossWeights(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/raw/stack/bc_large_aligned.ckpt")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--context", type=Path, default=ROOT / "data/processed/norman2019_eval.npz")
    parser.add_argument("--splits", type=Path, default=ROOT / "splits/norman2019_conditions.csv")
    parser.add_argument("--doubles", nargs="+", default=["all"], help="held-out doubles, a count, or 'all'")
    parser.add_argument("--steps", type=int, nargs="+", default=[1], help="generation steps T")
    parser.add_argument("--draws", type=int, default=2, help="generation draws per prompt order")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ttt-steps", type=int, default=50, help="adaptation steps; 0 runs ICL only")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--scope", default="final_block", choices=["final_block", "all_cross_cell"])
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--n-kept", type=int, default=128)
    parser.add_argument("--weights", default="", help='loss weights, e.g. "ed=1,recon=0,sw=0,cls=0"')
    parser.add_argument("--no-cls", action="store_true", help="do not co-adapt the unmasking classifier")
    parser.add_argument("--null", action="store_true", help="add the null control task")
    parser.add_argument("--null-adapter", action="store_true", help="also report an adapter trained on the null task only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ttt/norman_compare")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = EvalContext.load(args.context)
    split = pd.read_csv(args.splits)
    is_double = split["kind"].eq("combination")
    held_out = sorted(split.loc[split["split"].eq("test") & is_double, "condition"])
    train_doubles = sorted(split.loc[split["split"].eq("train") & is_double, "condition"])
    if args.doubles == ["all"]:
        doubles = held_out
    elif len(args.doubles) == 1 and args.doubles[0].isdigit():
        doubles = held_out[: int(args.doubles[0])]
    else:
        doubles = args.doubles
    assert set(doubles) <= set(held_out), "only held-out doubles can be scored"
    near = {d: neighbours(d, train_doubles) for d in doubles}
    log(f"{len(doubles)} held-out doubles; neighbourhood sizes {({d: len(n) for d, n in near.items()})}")

    # Cells needed: the doubles' singles; for TTT also the neighbours, their singles and control.
    needed = {g for d in doubles for g in d.split("+")} | {CONTROL}  # control: the diff reference
    if args.ttt_steps:
        needed |= {c for d in doubles for nb in near[d] for c in [nb, *nb.split("+")]}
    vocab = load_stack_genes(args.vocab)
    cells = load_norman(args.raw, vocab, conditions=needed)
    singles = sorted(c for c in set(cells.condition) if "+" not in c and c != CONTROL)
    pools = single_pools(cells.condition, singles, seed=args.seed) | control_pools(cells.condition, seed=args.seed)
    reference = pools[(CONTROL, "infer")]
    panel = list(ctx.truth.columns)
    log(f"loaded {cells.counts.shape[0]} cells from {len(needed)} conditions")

    model, hparams = load_stack(args.checkpoint, device=device)
    log(f"STACK on {device}: {hparams['model_config']}")
    if args.ttt_steps or args.null_adapter:
        inject_lora(model, stack_targets(model, args.scope), rank=args.rank, alpha=args.alpha)
    cls_base = {k: v.clone() for k, v in model.cls.state_dict().items()}

    predictions, costs, histories = {}, {}, {}

    def predict_all(prefix):
        for steps in args.steps:
            for double in doubles:
                tic = time.time()
                out = predict_double(model, cells, pools, double, args.vocab, panel, steps, args.draws,
                                     args.batch_size, reference=reference)
                for key, profile in out.items():
                    name = key if key.startswith("query") else f"{prefix}_T{steps}_{key}"
                    predictions.setdefault(name, {})[double] = profile
                costs.setdefault(double, {})[f"{prefix}_T{steps}_seconds"] = time.time() - tic
            log(f"{prefix} T={steps} done")

    def log_losses(label, history):
        by_kind = {}
        for h in history:
            task = h["task"]
            kind = "own prompts" if task.query is None else ("null" if task.target == CONTROL else "neighbours")
            by_kind.setdefault(kind, []).append(h)
        parts = []
        for kind, rows in by_kind.items():
            third = max(1, len(rows) // 3)
            key = "recon_loss" if kind == "own prompts" else "mmd_loss"
            first = sum(r[key] for r in rows[:third]) / third
            last = sum(r[key] for r in rows[-third:]) / third
            parts.append(f"{kind} {key.split('_')[0]} {first:.2f}->{last:.2f}")
        log(f"{label}: " + "; ".join(parts))

    with adapters_disabled(model):  # frozen STACK first, before any adapter has been trained
        predict_all("icl")

    weights = parse_weights(args.weights)
    config = AdaptConfig(steps=args.ttt_steps or 50, lr=args.lr, n_cells=args.n_cells, n_kept=args.n_kept,
                         weights=weights, train_cls=not args.no_cls, seed=args.seed)
    if args.null_adapter:
        result = adapt(model, cells, [Task(CONTROL, CONTROL, CONTROL)], pools, config, cls_base=cls_base, device=device)
        histories["nulladapt"] = [{**h, "task": str(h["task"])} for h in result["history"]]
        log_losses("null-only adapter", result["history"])
        predict_all("nulladapt")

    if args.ttt_steps:
        for double in doubles:
            tasks = norman_tasks(double, train_doubles, null=args.null)
            result = adapt(model, cells, tasks, pools, config, cls_base=cls_base, device=device)
            histories[double] = [{**h, "task": str(h["task"])} for h in result["history"]]
            costs[double].update({f"ttt_adapt_{k}": v for k, v in result.items() if k != "history"})
            costs[double]["ttt_tasks"] = len(tasks)
            log_losses(f"TTT {double} ({len(tasks)} tasks, {result['seconds']:.0f}s)", result["history"])
            for steps in args.steps:
                tic = time.time()
                out = predict_double(model, cells, pools, double, args.vocab, panel, steps, args.draws,
                                     args.batch_size, reference=reference)
                for key, profile in out.items():
                    if not key.startswith("query"):
                        predictions.setdefault(f"ttt_T{steps}_{key}", {})[double] = profile
                costs[double][f"ttt_T{steps}_seconds"] = time.time() - tic

    frames = {name: pd.DataFrame(p).T.loc[doubles, panel] for name, p in predictions.items()}
    long = ctx.score(frames, perturbations=doubles)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", models=np.array(list(frames), dtype=str),
                        perturbations=np.array(doubles, dtype=str), genes=np.array(panel, dtype=str),
                        values=np.stack([f.to_numpy(dtype=np.float32) for f in frames.values()]))
    long.to_csv(args.out / "scores.csv", index=False)
    (args.out / "costs.json").write_text(json.dumps(costs, indent=1))
    (args.out / "histories.json").write_text(json.dumps(histories))
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    shown = long[~long["model"].str.endswith("_one")]  # every row, including _one, is in scores.csv
    values, gaps = median_tables(shown, "value"), median_tables(shown, "gap_closed")
    summary = [
        "# TTT vs ICL for STACK on Norman held-out doubles\n",
        f"{len(doubles)} doubles; neighbourhood sizes: {', '.join(f'{d} {len(near[d])}' for d in doubles)}. "
        f"{args.draws} draws per prompt order. Settings in args.json, per-double cost in costs.json.\n",
    ]
    for panel_name in values:
        summary += [
            f"## Panel `{panel_name}`\n",
            "Median score:\n",
            markdown_table(values[panel_name]) + "\n",
            "Median share of the gap from each metric's negative control to a perfect score:\n",
            markdown_table(gaps[panel_name]) + "\n",
        ]
    (args.out / "summary.md").write_text("\n".join(summary))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
