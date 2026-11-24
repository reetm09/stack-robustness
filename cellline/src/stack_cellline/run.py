"""Fine-tune STACK on X-Atlas/Orion for Study 1, from the command line.

Loads the planned cells (stack_cellline.xatlas), fine-tunes (stack_cellline.finetune) and
writes adapter.pt, history.jsonl, val.json, run.json and plan.json to --out. Run
--plan-only first: it reports the genes, cells and batch files a run would use without
reading any expression data. See README.md for a first-run command.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .xatlas import Subset, load_xatlas


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="Stack-Large-Aligned .ckpt")
    p.add_argument("--vocab", type=Path, required=True, help="basecount_1000per_15000max.pkl")
    p.add_argument("--gene-metadata", type=Path, required=True, help="X-Atlas/Orion metadata/gene_metadata.parquet")
    p.add_argument("--obs", nargs="+", required=True, help="LINE=PATH per-cell metadata (scripts/fetch_xatlas_obs.py)")
    p.add_argument("--split", type=Path, required=True, help="split directory (genes.csv, controls.csv.gz, split.json)")
    p.add_argument("--data", required=True,
                   help="directory of <LINE>_Batch<N>.parquet, or hf://datasets/Xaira-Therapeutics/X-Atlas-Orion/data")
    p.add_argument("--source-line", default="HEK293T")
    p.add_argument("--target-line", default="HCT116")
    p.add_argument("--hgnc", type=Path, help="HGNC hgnc_complete_set.txt, to map renamed genes into the vocabulary")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true", help="report what would be loaded, then stop")
    g = p.add_argument_group("subset (default: everything; start small)")
    g.add_argument("--train-genes", type=int)
    g.add_argument("--val-genes", type=int)
    g.add_argument("--max-cells-per-gene", type=int, help="per gene and line")
    g.add_argument("--max-controls", type=int, help="per control pool")
    g.add_argument("--batches-per-line", type=int, help="batch files read per line")
    g.add_argument("--subset-seed", type=int, default=0)
    t = p.add_argument_group("training (defaults: stack_cellline.finetune.FinetuneConfig)")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    t.add_argument("--warmup", type=float, default=0.1)
    t.add_argument("--scope", default="final_block", choices=["final_block", "all_cross_cell"])
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--alpha", type=float, default=16.0)
    t.add_argument("--n-cells", type=int, default=512, help="cells per window")
    t.add_argument("--n-kept", type=int, default=128, help="prompt cells per window")
    t.add_argument("--reveal", default="target", choices=["target", "prompt"],
                   help="what revealed window positions show (stack_ttt.windows.task_window)")
    t.add_argument("--freeze-cls", action="store_true", help="do not train the unmasking classifier f_CLS")
    t.add_argument("--val-windows", type=int, default=2, help="fixed windows per validation gene")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = p.parse_args(argv)

    from stack_robustness.genes import load_stack_genes, vocabulary_aliases

    vocab = load_stack_genes(args.vocab)
    gene_metadata = pd.read_parquet(args.gene_metadata)
    aliases = None
    if args.hgnc:
        aliases = vocabulary_aliases(gene_metadata["gene_name"].astype(str).tolist(),
                                     gene_metadata["ensembl_id"].astype(str).tolist(), vocab, args.hgnc)
    subset = Subset(args.train_genes, args.val_genes, args.max_cells_per_gene, args.max_controls,
                    args.batches_per_line, args.subset_seed)
    loaded = load_xatlas(args.obs, args.split, args.data, gene_metadata, vocab, args.source_line,
                         args.target_line, subset, aliases=aliases, plan_only=args.plan_only)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"subset": asdict(subset), "aliases": len(aliases or {}), **loaded["report"]}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    if args.plan_only:
        return plan

    import torch
    from stack_ttt.adapt import AdaptConfig

    from .finetune import FinetuneConfig, finetune

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = FinetuneConfig(
        scope=args.scope, rank=args.rank, alpha=args.alpha, val_windows=args.val_windows, val_seed=args.seed,
        adapt=AdaptConfig(steps=args.steps, lr=args.lr, weight_decay=args.weight_decay, schedule=args.schedule,
                          warmup=args.warmup, n_cells=args.n_cells, n_kept=args.n_kept, reveal=args.reveal,
                          train_cls=not args.freeze_cls, seed=args.seed),
    )
    result = finetune(args.checkpoint, loaded["data"], loaded["split"], args.out, config, device=device)
    return {**plan, "val_loss_before": result.val_before["loss"], "val_loss_after": result.val_after["loss"]}


if __name__ == "__main__":
    main()
