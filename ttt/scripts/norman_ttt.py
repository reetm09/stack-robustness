"""Study 4 on Norman: in-context learning, a shared adapter, and per-pair test-time training.

For each held-out double A+B, the arms are {frozen, shared} x {no TTT, TTT on neighbours,
TTT on random pairs}:
  icl                frozen STACK, prompted with A's cells and queried with B's, and the reverse
  shared             one adapter trained on every training double, in both orders, plus the
                     null task: Study 2's fine-tuned STACK and the spec's shared adapter.
                     Akyurek et al. start TTT from such a task-format fine-tuned model.
  ttt                a fresh adapter trained on A+B's tasks: the training doubles sharing a gene
                     with A+B (both orders), the null task, masked reconstruction of A's and B's
                     own cells
  ttt_random         the same, with as many random training doubles sharing no gene with A+B
  shared_ttt         ttt, starting from the shared adapter (reset to it after each pair)
  shared_ttt_random  ttt_random, starting from the shared adapter

--holdout val scores the validation doubles that scripts/norman_baselines.py --validation
carved out of the training doubles, and trains on the rest; --holdout test scores the 26 test
doubles. Single-perturbation cells are split into "adapt" and "infer" halves; training draws
on "adapt" and predictions start from "infer". Each pair is saved when it finishes
(pairs/<double>.npz), so rerunning the same command resumes.

Adaptation windows reveal more prompt cells after the prompt, as STACK's inference does
(--reveal prompt; "target" reveals true target cells, as STACK's post-training and version 1
of this study did). f_CLS stays frozen unless --train-cls. Norman genes that HGNC has renamed
are mapped into the vocabulary slots they were missing from, unless --no-aliases.

In the validation run, training the shared adapter also records, every 200 steps, its loss on
fixed windows of the validation doubles and of as many training doubles (shared_history.json).
This is a diagnostic for the report's loss curves, not a selection rule (PROTOCOL.md). It is
the only use of the validation doubles' cells; the test run never loads the test doubles.

    uv run python scripts/norman_ttt.py --holdout val --arms shared --out ../results/ttt/val/shared
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from stack_robustness.evaluation import EvalContext, markdown_table, median_tables
from stack_robustness.genes import load_stack_genes, vocabulary_aliases
from stack_ttt.adapt import AdaptConfig, adapt, task_losses
from stack_ttt.lora import adapter_state, adapters_disabled, inject_lora, load_adapter_state, stack_targets
from stack_ttt.model import load_stack
from stack_ttt.norman import load_norman
from stack_ttt.predict import predict_double
from stack_ttt.tasks import CONTROL, cell_pools, neighbours, norman_tasks, random_tasks, shared_tasks

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["icl", "shared", "ttt", "ttt_random", "shared_ttt", "shared_ttt_random"]
START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START:6.0f}s] {message}", flush=True)


def loss_summary(history) -> dict:
    """Mean loss over the first and last fifth of the steps, by kind of task."""
    by_kind = {}
    for h in history:
        task = h["task"]
        kind = "own_prompts" if task.query is None else ("null" if task.target == CONTROL else "pairs")
        by_kind.setdefault(kind, []).append(h["loss"])
    return {k: [float(np.mean(v[: max(1, len(v) // 5)])), float(np.mean(v[-max(1, len(v) // 5):]))]
            for k, v in by_kind.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--holdout", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/raw/stack/bc_large_aligned.ckpt")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/raw/stack_vocab/basecount_1000per_15000max.pkl")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/norman/NormanWeissman2019_filtered.h5ad")
    parser.add_argument("--doubles", nargs="+", default=["all"], help="held-out doubles, a count, or 'all'")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--steps", type=int, nargs="+", default=[1], help="generation steps T")
    parser.add_argument("--random-steps", type=int, nargs="+", default=[1], help="T for the random-pair arms")
    parser.add_argument("--draws", type=int, default=2, help="generation draws per prompt order")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scope", default="final_block", choices=["final_block", "all_cross_cell"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--shared-steps", type=int, default=2000)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--shared-adapter", type=Path, default=None,
                        help="load (or save) the shared adapter here; default <out>/shared_adapter.pt")
    parser.add_argument("--ttt-steps", type=int, default=250)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--selection", type=Path, default=None,
                        help="stage2_selection.json (scripts/select_ttt.py): each TTT arm's own learning rate and "
                             "steps, in place of --ttt-lr and --ttt-steps")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--no-null", action="store_true", help="leave out the null control task")
    parser.add_argument("--reveal", default="prompt", choices=["prompt", "target"],
                        help="what adaptation windows reveal after the prompt (version 1: target)")
    parser.add_argument("--train-cls", action="store_true", help="co-adapt f_CLS (version 1 did)")
    parser.add_argument("--no-aliases", action="store_true", help="match genes by symbol only, as Arc's code does")
    parser.add_argument("--hgnc", type=Path, default=ROOT / "data/raw/hgnc/hgnc_complete_set.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    null = not args.no_null

    per_arm = {}  # TTT arm -> (steps, learning rate) chosen on validation
    if args.selection is not None:
        per_arm = {arm: (int(v["steps"]), float(v["lr"])) for arm, v in json.loads(args.selection.read_text()).items()}

    name = "norman2019_val" if args.holdout == "val" else "norman2019"
    ctx = EvalContext.load(ROOT / f"data/processed/{name}_eval.npz")
    split = pd.read_csv(ROOT / f"splits/{name}_conditions.csv")
    is_double = split["kind"].eq("combination")
    held_out = sorted(split.loc[split["split"].eq(args.holdout) & is_double, "condition"])
    train_doubles = sorted(split.loc[split["split"].eq("train") & is_double, "condition"])
    if args.doubles == ["all"]:
        doubles = held_out
    elif len(args.doubles) == 1 and args.doubles[0].isdigit():
        doubles = held_out[: int(args.doubles[0])]
    else:
        doubles = args.doubles
    assert set(doubles) <= set(held_out), "only held-out doubles can be scored"
    log(f"{args.holdout}: {len(doubles)} held-out doubles, {len(train_doubles)} training doubles; "
        f"neighbourhood sizes {[len(neighbours(d, train_doubles)) for d in doubles]}")

    singles = sorted(split.loc[split["kind"].eq("single"), "condition"])
    vocab = load_stack_genes(args.vocab)
    aliases = {}
    if not args.no_aliases:
        var = ad.read_h5ad(args.raw, backed="r").var
        aliases = vocabulary_aliases(list(map(str, var.index)), list(var["ensemble_id"].astype(str)), vocab, args.hgnc)
    monitored = held_out if args.holdout == "val" else []  # only for the validation loss curve
    cells = load_norman(args.raw, vocab, conditions=set(singles) | set(train_doubles) | {CONTROL} | set(monitored),
                        aliases=aliases)
    assert not set(cells.condition) & (set(held_out) - set(monitored)), "test doubles must never be loaded"
    pools = cell_pools(cells.condition, seed=args.seed)
    assert not {key[0] for key in pools} & set(held_out), "held-out cells must never be pooled"
    reference = pools[(CONTROL, "infer")]
    panel = list(ctx.truth.columns)
    log(f"loaded {cells.counts.shape[0]} cells; {len(aliases)} genes renamed into the vocabulary")

    model, _ = load_stack(args.checkpoint, device=device)
    inject_lora(model, stack_targets(model, args.scope), rank=args.rank, alpha=args.alpha)
    cls_frozen = {k: v.clone() for k, v in model.cls.state_dict().items()}

    def config(steps, lr):
        return AdaptConfig(steps=steps, lr=lr, weight_decay=args.weight_decay, schedule=args.schedule,
                           warmup=args.warmup, seed=args.seed, reveal=args.reveal, train_cls=args.train_cls)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs").mkdir(exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    shared, cls_shared = None, None
    if any(arm.startswith("shared") for arm in args.arms):
        path = args.shared_adapter or args.out / "shared_adapter.pt"
        settings = {"scope": args.scope, "rank": args.rank, "alpha": args.alpha, "holdout": args.holdout,
                    "null": null, "aliases": not args.no_aliases, **asdict(config(args.shared_steps, args.shared_lr))}
        settings["weights"] = str(settings["weights"])
        if path.exists():
            saved = torch.load(path, map_location=device, weights_only=True)
            assert saved["settings"] == settings, f"{path} was trained with other settings: {saved['settings']}"
            shared, cls_shared = saved["adapters"], saved["cls"]
            log(f"loaded the shared adapter from {path}")
        else:
            tasks = shared_tasks(train_doubles, null=null)
            assert not {t.target for t in tasks} & set(held_out), "held-out doubles must never be trained on"
            shared_config = config(args.shared_steps, args.shared_lr)
            monitor = None
            if monitored:
                probe = sorted(np.random.default_rng(args.seed).choice(train_doubles, len(monitored), replace=False).tolist())
                probes = {"train": shared_tasks(probe), "val": shared_tasks(monitored)}

                def monitor(m):
                    return {f"{name}_{k}": v for name, probe_tasks in probes.items()
                            for k, v in task_losses(m, cells, probe_tasks, pools, shared_config, device).items()}

            result = adapt(model, cells, tasks, pools, shared_config, cls_base=cls_frozen, device=device,
                           monitor=monitor, monitor_every=200)
            shared = adapter_state(model)
            cls_shared = {k: v.clone() for k, v in model.cls.state_dict().items()}
            torch.save({"adapters": shared, "cls": cls_shared, "settings": settings}, path)
            (args.out / "shared_history.json").write_text(json.dumps(
                {"seconds": result["seconds"], "loss": [h["loss"] for h in result["history"]],
                 "lr": [h["lr"] for h in result["history"]],
                 "monitor": result["monitor"], "monitor_seconds": result["monitor_seconds"]}))
            log(f"trained the shared adapter on {len(tasks)} tasks in {result['seconds']:.0f}s; "
                f"loss {loss_summary(result['history'])}")
            if result["monitor"]:
                first, last = result["monitor"][0], result["monitor"][-1]
                log(f"loss on fixed windows, step 0 -> {last['step']}: validation doubles "
                    f"{first['val_loss']:.3f} -> {last['val_loss']:.3f}, training doubles "
                    f"{first['train_loss']:.3f} -> {last['train_loss']:.3f} ({result['monitor_seconds']:.0f}s)")

    for double in doubles:
        target = args.out / "pairs" / f"{double}.npz"
        if target.exists():
            continue
        profiles, info = {}, {"seconds": {}, "losses": {}, "tasks": {}}

        def predict(arm, steps_list):
            for steps in steps_list:
                tic = time.time()
                out = predict_double(model, cells, pools, double, args.vocab, panel, steps, args.draws,
                                     args.batch_size, reference=reference)
                for key, profile in out.items():
                    profiles[key if key.startswith("query") else f"{arm}_T{steps}_{key}"] = profile
                info["seconds"][f"{arm}_T{steps}"] = time.time() - tic

        for arm in args.arms:
            steps_list = args.random_steps if arm.endswith("random") else args.steps
            if arm == "icl":
                model.cls.load_state_dict(cls_frozen)
                with adapters_disabled(model):
                    predict(arm, steps_list)
                continue
            if arm == "shared":
                load_adapter_state(model, shared)
                model.cls.load_state_dict(cls_shared)
                predict(arm, steps_list)
                continue
            if arm.endswith("random"):
                tasks = random_tasks(double, train_doubles, seed=args.seed, null=null)
            else:
                tasks = norman_tasks(double, train_doubles, null=null)
            assert not {t.target for t in tasks} & set(held_out), "held-out doubles must never be trained on"
            start = shared if arm.startswith("shared") else None
            steps_lr = per_arm.get(arm, (args.ttt_steps, args.ttt_lr))
            info.setdefault("ttt_settings", {})[arm] = {"steps": steps_lr[0], "lr": steps_lr[1]}
            result = adapt(model, cells, tasks, pools, config(*steps_lr),
                           cls_base=cls_shared if start is not None else cls_frozen, device=device,
                           adapter_init=start)
            info["seconds"][f"{arm}_adapt"] = result["seconds"]
            info["losses"][arm] = loss_summary(result["history"])
            info["tasks"][arm] = sorted({t.target for t in tasks if t.target not in (None, CONTROL)})
            predict(arm, steps_list)

        names = sorted(profiles)
        np.savez_compressed(target, names=np.array(names, dtype=str), genes=np.array(panel, dtype=str),
                            values=np.stack([profiles[n].reindex(panel).to_numpy(np.float32) for n in names]))
        (args.out / "pairs" / f"{double}.json").write_text(json.dumps(info, indent=1))
        log(f"{double}: {', '.join(f'{k} {v:.0f}s' for k, v in info['seconds'].items())}")

    # Score every finished pair together, beside the baselines, on the same genes.
    frames = {}
    for double in doubles:
        saved = np.load(args.out / "pairs" / f"{double}.npz")
        for model_name, row in zip(saved["names"], saved["values"]):
            frames.setdefault(str(model_name), {})[double] = pd.Series(row, index=saved["genes"])
    frames = {k: pd.DataFrame(v).T.loc[doubles, panel] for k, v in frames.items()}
    long = ctx.score(frames, perturbations=doubles)
    long.to_csv(args.out / "scores.csv", index=False)
    shown = long[~long["model"].str.endswith("_one")]
    values, gaps = median_tables(shown, "value"), median_tables(shown, "gap_closed")
    lines = [f"# Norman {args.holdout} doubles: ICL, shared adapter, TTT\n",
             f"{len(doubles)} doubles, {len(train_doubles)} training doubles. Settings in args.json.\n"]
    for panel_name in values:
        lines += [f"## Panel `{panel_name}`\n", "Median score:\n", markdown_table(values[panel_name]) + "\n",
                  "Median share of the gap closed:\n", markdown_table(gaps[panel_name]) + "\n"]
    (args.out / "summary.md").write_text("\n".join(lines))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
