"""Minimal low-rank adapters (LoRA, Hu et al. 2022) for STACK's linear layers."""

import math
from contextlib import contextmanager

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable update scaling * B @ A, with B starting at zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.enabled = True
        kwargs = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **kwargs))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, rank, **kwargs))
        self.reset()

    def reset(self) -> None:
        """Fresh adapter: random A, zero B, so the layer equals its base again."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out


def stack_targets(model: nn.Module, scope: str) -> list:
    """Names of the linear layers to adapt in a STACK model.

    "final_block": the last layer's cross-cell attention (qkv, proj) and the decoder's
    two linears. The last layer's within-cell attention and token MLP are only
    token_dim wide, so they are left alone.
    "all_cross_cell": cross-cell attention in every layer, plus the decoder.
    """
    decoder = ["output_mlp.0", "output_mlp.3"]
    if scope == "final_block":
        layers = [len(model.layers) - 1]
    elif scope == "all_cross_cell":
        layers = range(len(model.layers))
    else:
        raise ValueError(f"unknown adapter scope {scope!r}")
    return [f"layers.{i}.gene_attn.{part}" for i in layers for part in ("qkv", "proj")] + decoder


def inject_lora(model: nn.Module, targets, rank: int, alpha: float) -> list:
    """Freeze `model` and wrap each named nn.Linear in a LoRALinear, in place."""
    model.requires_grad_(False)
    wrapped = []
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        layer = parent.get_submodule(child)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is {type(layer).__name__}, not nn.Linear")
        parent._modules[child] = LoRALinear(layer, rank, alpha)
        wrapped.append(parent._modules[child])
    return wrapped


def lora_modules(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def adapter_state(model: nn.Module) -> list:
    """A copy of every adapter's (A, B), in module order, for load_adapter_state."""
    return [(layer.lora_A.detach().clone(), layer.lora_B.detach().clone()) for layer in lora_modules(model)]


def load_adapter_state(model: nn.Module, state) -> None:
    layers = lora_modules(model)
    if len(layers) != len(state):
        raise ValueError(f"{len(state)} saved adapters for {len(layers)} in the model")
    with torch.no_grad():
        for layer, (a, b) in zip(layers, state):
            layer.lora_A.copy_(a)
            layer.lora_B.copy_(b)


@contextmanager
def adapters_disabled(model: nn.Module):
    """Run the base model (e.g. as the frozen teacher) with every adapter switched off."""
    layers = lora_modules(model)
    states = [layer.enabled for layer in layers]
    for layer in layers:
        layer.enabled = False
    try:
        yield model
    finally:
        for layer, state in zip(layers, states):
            layer.enabled = state
