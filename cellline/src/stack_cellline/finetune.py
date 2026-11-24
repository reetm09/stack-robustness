"""Fine-tune STACK on a target cell line (Study 1) and save the adapters.

Python API only. The caller loads the cells (CellLineData) and the split (Split); a file
loader and a command line wait for the team to confirm the X-Atlas/Orion schema and the
split format. See README.md.

Training reuses stack_ttt unchanged: LoRA adapters on the final block's cross-cell
attention and the decoder ("final_block"), plus the unmasking classifier f_CLS, trained by
stack_ttt.adapt.adapt with STACK's fine-tuning loss and a frozen teacher (the base model,
adapters off). Only the training genes and the training control pool enter training.

Validation loss is measured on fixed windows of the validation genes, with the validation
control pool as queries, before training (the adapters start at zero, so this is zero-shot
STACK) and after. adapt() has no per-step hook, so there is no validation during a run and
no choice among a run's steps. Settings (steps, learning rate) are compared across runs by
their validation loss, as in ttt/PROTOCOL.md.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from stack_ttt.adapt import COMPONENTS, AdaptConfig, adapt
from stack_ttt.lora import inject_lora, stack_targets
from stack_ttt.model import load_stack
from stack_ttt.objective import teacher_embeddings, ttt_loss
from stack_ttt.windows import task_window

from .artifact import save_adapter
from .tasks import CellLineData, Split, training_tasks, validation_rows


def _default_adapt() -> AdaptConfig:
    # The shared adapter's defaults in ttt/scripts/norman_ttt.py. Not tuned for any cell line.
    return AdaptConfig(steps=2000, lr=1e-4, weight_decay=0.01, schedule="cosine", warmup=0.1, seed=0)


@dataclass(frozen=True)
class FinetuneConfig:
    scope: str = "final_block"  # or "all_cross_cell" (stack_ttt.lora.stack_targets)
    rank: int = 8
    alpha: float = 16.0
    adapt: AdaptConfig = field(default_factory=_default_adapt)
    val_windows: int = 2  # fixed windows per validation gene
    val_seed: int = 0


@dataclass
class FinetuneResult:
    model: torch.nn.Module  # in eval mode, adapters loaded
    adapter_path: Path
    val_before: dict
    val_after: dict
    history: list
    seconds: float


def validation_loss(model, counts, rows: dict, config: FinetuneConfig, device="cpu") -> dict:
    """Mean loss components over fixed windows of each validation gene, then over genes.

    Each call draws the same windows and the same masking and context draws, inside a forked
    torch random state, so it neither depends on nor disturbs the training draws. The model is
    evaluated with every parameter frozen, as load_finetuned returns it, so the loss before
    training equals zero-shot STACK's; the trainable flags are restored afterwards.
    """
    a = config.adapt
    was_training = model.training
    trainable = [p.requires_grad for p in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    try:
        return _validation_loss(model, counts, rows, config, a, device)
    finally:
        for p, flag in zip(model.parameters(), trainable):
            p.requires_grad_(flag)
        model.train(was_training)


def _validation_loss(model, counts, rows, config, a, device) -> dict:
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    per_gene = {}
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for i, (gene, (prompt, query, target)) in enumerate(rows.items()):
            values = []
            for w in range(config.val_windows):
                seeds = np.random.SeedSequence([config.val_seed, i, w])
                rng = np.random.default_rng(seeds)
                # Same window layout as training (stack_ttt.adapt._window_loss), including what the
                # revealed positions show (`a.reveal`: target cells, or more prompt cells).
                observed, truth, revealed = task_window(counts, prompt, query, target, a.n_cells, a.n_kept, rng,
                                                        reveal=a.reveal)
                observed = torch.from_numpy(observed)[None].to(device)
                truth = torch.from_numpy(truth)[None].to(device)
                revealed = None if revealed is None else torch.from_numpy(revealed)[None].to(device)
                torch.manual_seed(int(seeds.generate_state(1)[0]))
                out = ttt_loss(model, observed, truth, a.n_kept, teacher_embeddings(model, truth), a.weights,
                               revealed=revealed)
                values.append({k: float(out[k]) for k in COMPONENTS})
            per_gene[gene] = {k: float(np.mean([v[k] for v in values])) for k in COMPONENTS}
    mean = {k: float(np.mean([g[k] for g in per_gene.values()])) for k in COMPONENTS}
    return {**mean, "per_gene": per_gene}


def finetune(checkpoint, data: CellLineData, split: Split, out_dir, config: FinetuneConfig = FinetuneConfig(),
             device="cpu") -> FinetuneResult:
    """Train adapters on the training genes and write to `out_dir`:
      adapter.pt      the adapters and f_CLS (stack_cellline.artifact; load with load_finetuned)
      history.jsonl   per training step: the task, learning rate and loss components
      val.json        validation loss before (zero-shot) and after training, per gene and mean
      run.json        the settings, the cost and the mean validation losses
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells, pools, tasks = training_tasks(data, split)
    rows = validation_rows(data, split)
    _log(f"{len(tasks)} training genes, {len(rows)} validation genes, "
         f"{len(split.train_controls)}/{len(split.val_controls)} training/validation control cells")

    model, _ = load_stack(checkpoint, device=device)
    targets = stack_targets(model, config.scope)
    inject_lora(model, targets, rank=config.rank, alpha=config.alpha)
    cls_frozen = {k: v.clone() for k, v in model.cls.state_dict().items()}

    val_before = validation_loss(model, cells.counts, rows, config, device)
    _log(f"validation loss before training (zero-shot): {val_before['loss']:.4f}")
    tic = time.time()
    trained = adapt(model, cells, tasks, pools, config.adapt, cls_base=cls_frozen, device=device)
    seconds = time.time() - tic
    val_after = validation_loss(model, cells.counts, rows, config, device)
    _log(f"trained {config.adapt.steps} steps in {seconds:.0f}s; validation loss after: {val_after['loss']:.4f}")

    settings = {
        "base_checkpoint": str(checkpoint),
        "targets": list(targets),
        "scope": config.scope,
        "rank": config.rank,
        "alpha": config.alpha,
        "adapt": asdict(config.adapt),
        "val_windows": config.val_windows,
        "val_seed": config.val_seed,
        "source_line": str(split.source_line),
        "target_line": str(split.target_line),
        "train_genes": sorted(str(g) for g in split.train_genes),
        "val_genes": sorted(str(g) for g in split.val_genes),
        "n_train_controls": int(len(split.train_controls)),
        "n_val_controls": int(len(split.val_controls)),
    }
    adapter_path = out / "adapter.pt"
    save_adapter(adapter_path, model, settings)

    history = [
        {"step": i, "prompt": h["task"].prompt, "target": h["task"].target, "lr": h["lr"],
         **{k: h[k] for k in COMPONENTS}}
        for i, h in enumerate(trained["history"])
    ]
    (out / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history))
    (out / "val.json").write_text(json.dumps({"before": val_before, "after": val_after}, indent=1))
    cost = {k: trained[k] for k in ("seconds", "peak_gpu_gb") if k in trained}
    summary = {"settings": settings, "cost": cost,
               "val_loss_before": val_before["loss"], "val_loss_after": val_after["loss"]}
    (out / "run.json").write_text(json.dumps(summary, indent=1))
    _log(f"wrote {out}")
    return FinetuneResult(model=model, adapter_path=adapter_path, val_before=val_before, val_after=val_after,
                          history=history, seconds=seconds)


def _log(message: str) -> None:
    print(f"[cellline] {message}", flush=True)
