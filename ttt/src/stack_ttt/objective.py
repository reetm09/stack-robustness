"""The test-time training objective: STACK's fine-tuning loss, reweighted.

Reassembled from ICL_FinetunedModel's own helpers (arc-stack 0.1.3) for two reasons:
the components can be weighted separately, and a window may hold prompt cells only, so
the test instance's own prompt cells can train masked reconstruction. Upstream's forward
skips reconstruction when nothing is being predicted. For a window with query positions
and default weights, every component equals upstream's forward under the same seed.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .lora import adapters_disabled


@dataclass(frozen=True)
class LossWeights:
    ed: float = 1.0  # energy distance in gene space and teacher-embedding space (upstream "mmd_loss")
    recon: float = 1.0  # negative-binomial reconstruction of masked genes
    sw: Optional[float] = None  # sliced Wasserstein to N(0, I); None uses the model's sw_weight
    cls: float = 1.0  # the unmasking classifier f_CLS


@torch.no_grad()
def teacher_embeddings(model, truth: torch.Tensor) -> torch.Tensor:
    """Frozen-teacher embeddings of the target cells: the base model with adapters off,
    every position a fully observed prompt cell, as in STACK's fine-tuning."""
    with adapters_disabled(model):
        out = model(
            observed_features=truth,
            ground_truth_features=truth,
            mask_genes=False,
            n_kept_cell=truth.shape[1],
            return_loss=False,
        )
    return out["final_cell_embeddings"].detach()


def ttt_loss(model, observed, truth, n_kept: int, teacher, weights: LossWeights = LossWeights(), revealed=None) -> dict:
    """Loss on one window of counts (batch x cells x genes).

    Positions [0, n_kept) are prompt cells (observed == truth). The rest are query cells:
    `observed` holds the cells to transform and `truth` the target cells. A random share
    of query positions, right after the prompt, is revealed as context, as in STACK's
    fine-tuning: they show their target cells, or, given `revealed`, its cells there, which
    then also stand in for the targets of those positions (input, library and
    reconstruction alike). The teacher still sees the true window.
    """
    batch, n_cells, n_genes = observed.shape
    device = observed.device
    zero = torch.zeros((), device=device)

    features_log = torch.log1p(observed)
    time = torch.rand(1, device=device).item() * 0.9375
    n_context = n_kept + int((n_cells - n_kept) * time)
    if revealed is not None:
        truth = truth.clone()
        truth[:, n_kept:n_context] = revealed[:, n_kept:n_context]
    features_log[:, n_kept:n_context] = torch.log1p(truth[:, n_kept:n_context])
    masked, mask = model.apply_finetune_mask(features_log)
    x = model._reduce_and_tokenize(masked)

    has_query = n_kept < n_cells
    attn_mask = torch.zeros(n_cells, n_cells, dtype=torch.bool, device=device)
    if has_query:
        to_predict = torch.zeros_like(x)
        to_predict[:, n_context:] = 1.0
        x = x + model.query_pos_embedding[None, None] * to_predict
        attn_mask[:n_kept, n_kept:] = True  # prompt cells cannot see query cells
    x = model._run_attention_layers(x, gene_attn_mask=attn_mask[None, None])
    emb = x.reshape(batch, n_cells, -1)
    lib = truth.sum(dim=-1, keepdim=True).clamp(min=1.0)
    nb_mean, nb_disp, _ = model._compute_nb_parameters(emb, lib)

    if has_query:
        cls_loss, _ = model._compute_cls_loss(
            emb[:, :n_kept].mean(dim=1).detach(), emb[:, n_kept:n_context].detach(), emb[:, n_context:].detach()
        )
    else:
        cls_loss = zero
    if mask[:, :n_context].any():
        recon_loss, _ = model._compute_reconstruction_loss(
            nb_mean[:, :n_context], nb_disp[:, :n_context], truth[:, :n_context], mask[:, :n_context]
        )
    else:
        recon_loss = zero
    mmd_loss, _, _ = model._compute_mmd_loss(
        nb_mean, nb_disp, truth, lib, emb, teacher, None, None, n_context, n_cells, n_genes, time, device
    )
    k = max(32, min(128, n_cells))
    sw_loss = model._compute_sw_loss(emb, subsample_size=k, min_size=k, max_size=k, n_proj=getattr(model, "n_proj", None))

    sw_weight = model.sw_weight if weights.sw is None else weights.sw
    loss = weights.ed * mmd_loss + weights.recon * recon_loss + sw_weight * sw_loss + weights.cls * cls_loss
    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "mmd_loss": mmd_loss,
        "sw_loss": sw_loss,
        "cls_loss": cls_loss,
        "n_context": n_context,
    }
