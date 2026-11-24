"""Test-time adaptation: adapters trained on one held-out instance's tasks."""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .lora import load_adapter_state, lora_modules
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
    schedule: str = "constant"  # "cosine": linear warmup, then cosine decay to zero
    warmup: float = 0.0  # share of the steps spent warming up
    reveal: str = "target"  # what revealed positions show: "target" cells, or more "prompt" cells (windows.task_window)


def lr_factor(step: int, total: int, warmup_steps: int, schedule: str) -> float:
    """Learning-rate multiplier at `step`. "cosine" is torchtune's
    get_cosine_schedule_with_warmup, which Akyurek et al.'s TTT recipe uses."""
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown schedule {schedule!r}")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _window_loss(model, cells, task, pools, config: AdaptConfig, rng, device) -> dict:
    """The loss components on one window of `task`, drawn with `rng`."""
    prompt, query, target = task_rows(task, pools, cells.condition)
    kept = config.n_kept if query is not None else config.n_cells
    observed, truth, revealed = task_window(cells.counts, prompt, query, target, config.n_cells, kept, rng,
                                            reveal=config.reveal)
    observed = torch.from_numpy(observed)[None].to(device)
    truth = torch.from_numpy(truth)[None].to(device)
    revealed = None if revealed is None else torch.from_numpy(revealed)[None].to(device)
    teacher = teacher_embeddings(model, truth) if query is not None else None
    return ttt_loss(model, observed, truth, kept, teacher, config.weights, revealed=revealed)


def _devices(device) -> list:
    return [torch.cuda.current_device()] if str(device).startswith("cuda") else []


def task_losses(model, cells, tasks, pools, config: AdaptConfig, device="cpu") -> dict:
    """Mean loss components over one fixed window per task, without gradients. Every call
    draws the same windows and masks, so a curve of it follows the model alone. Torch's
    global generator is left as it was found."""
    totals = dict.fromkeys(COMPONENTS, 0.0)
    with torch.random.fork_rng(devices=_devices(device)), torch.no_grad():
        torch.manual_seed(config.seed)
        for i, task in enumerate(tasks):
            out = _window_loss(model, cells, task, pools, config, np.random.default_rng([config.seed, i]), device)
            for k in COMPONENTS:
                totals[k] += out[k].item() / len(tasks)
    return totals


def adapt(model, cells, tasks, pools, config: AdaptConfig, cls_base=None, device="cpu", adapter_init=None,
          monitor=None, monitor_every: int = 100) -> dict:
    """Start the adapters fresh (or from `adapter_init`, a saved lora.adapter_state) and f_CLS
    from `cls_base`, if given, then train them on windows drawn from `tasks`. The base model
    stays frozen. Returns the loss history and cost.

    `monitor(model)`, if given, returns a dict of measurements (e.g. task_losses on held-out
    tasks). It runs in eval mode before the first step, every `monitor_every` steps and after
    the last, and the training run is the same as without it."""
    torch.manual_seed(config.seed)  # before the reset: fresh adapters depend on the seed alone
    adapters = lora_modules(model)
    if adapter_init is None:
        for layer in adapters:
            layer.reset()
    else:
        load_adapter_state(model, adapter_init)
    if cls_base is not None:
        model.cls.load_state_dict(cls_base)
    params = [p for layer in adapters for p in (layer.lora_A, layer.lora_B)]
    model.cls.requires_grad_(config.train_cls)
    if config.train_cls:
        params += list(model.cls.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    warmup_steps = round(config.warmup * config.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, config.steps, warmup_steps, config.schedule)
    )
    rng = np.random.default_rng(config.seed)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    history, measured, tic, monitor_seconds = [], [], time.time(), 0.0

    def measure(step):
        nonlocal monitor_seconds
        start = time.time()
        model.eval()
        with torch.random.fork_rng(devices=_devices(device)):  # training draws the same numbers either way
            measured.append({"step": step} | monitor(model))
        model.train()
        monitor_seconds += time.time() - start

    model.train()
    for step in range(config.steps):
        if monitor is not None and step % monitor_every == 0:
            measure(step)
        task = tasks[rng.integers(len(tasks))]
        out = _window_loss(model, cells, task, pools, config, rng, device)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        scheduler.step()
        history.append({"task": task, "lr": lr} | {k: out[k].detach().item() for k in COMPONENTS})
    if monitor is not None:
        measure(config.steps)
    model.eval()

    cost = {"seconds": time.time() - tic - monitor_seconds, "monitor_seconds": monitor_seconds}
    if str(device).startswith("cuda"):
        cost["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return {"history": history, "monitor": measured, **cost}
