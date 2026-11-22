"""Test-time adaptation: fresh adapters trained on one held-out instance's tasks."""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .lora import lora_modules
from .objective import LossWeights, teacher_embeddings, ttt_loss
from .tasks import task_rows
from .windows import task_window

COMPONENTS = ("loss", "recon_loss", "mmd_loss", "sw_loss", "cls_loss")


@dataclass(frozen=True)
class AdaptConfig:
    steps: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_cells: int = 512  # window size; the checkpoint's own is 512
    n_kept: int = 128  # prompt cells per window; the checkpoint was fine-tuned with 128
    weights: LossWeights = field(default_factory=LossWeights)
    train_cls: bool = True  # co-adapt the unmasking classifier f_CLS
    seed: int = 0


def adapt(model, cells, tasks, pools, config: AdaptConfig, cls_base=None, device="cpu") -> dict:
    """Reset the adapters (and f_CLS to `cls_base`, if given), then train them on windows
    drawn from `tasks`. The base model stays frozen. Returns the loss history and cost."""
    adapters = lora_modules(model)
    for layer in adapters:
        layer.reset()
    if cls_base is not None:
        model.cls.load_state_dict(cls_base)
    params = [p for layer in adapters for p in (layer.lora_A, layer.lora_B)]
    model.cls.requires_grad_(config.train_cls)
    if config.train_cls:
        params += list(model.cls.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    history, tic = [], time.time()
    model.train()
    for _ in range(config.steps):
        task = tasks[rng.integers(len(tasks))]
        prompt, query, target = task_rows(task, pools, cells.condition)
        kept = config.n_kept if query is not None else config.n_cells
        observed, truth = task_window(cells.counts, prompt, query, target, config.n_cells, kept, rng)
        observed = torch.from_numpy(observed)[None].to(device)
        truth = torch.from_numpy(truth)[None].to(device)
        teacher = teacher_embeddings(model, truth) if query is not None else None
        out = ttt_loss(model, observed, truth, kept, teacher, config.weights)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        history.append({"task": task} | {k: out[k].detach().item() for k in COMPONENTS})
    model.eval()

    cost = {"seconds": time.time() - tic}
    if str(device).startswith("cuda"):
        cost["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return {"history": history, **cost}
