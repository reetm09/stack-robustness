import math

import pytest
import torch

from stack_ttt.lora import inject_lora, stack_targets
from stack_ttt.objective import LossWeights, teacher_embeddings, ttt_loss

N_KEPT = 6
COMPONENTS = ["recon_loss", "mmd_loss", "sw_loss", "cls_loss"]


def test_loss_components_match_stack_fine_tuning_forward(tiny_model, tiny_counts):
    observed, truth = tiny_counts
    teacher = teacher_embeddings(tiny_model, truth)

    torch.manual_seed(7)
    upstream = tiny_model(observed, truth, n_kept_cell=N_KEPT, t_cell_embeddings=teacher, return_loss=True)
    torch.manual_seed(7)
    ours = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher)

    for name in COMPONENTS:
        assert ours[name].item() == pytest.approx(upstream[name].item(), rel=1e-5), name
    assert ours["loss"].item() == pytest.approx(upstream["loss"].item(), rel=1e-5)


def test_weights_scale_each_component(tiny_model, tiny_counts):
    observed, truth = tiny_counts
    teacher = teacher_embeddings(tiny_model, truth)

    torch.manual_seed(3)
    full = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher)
    torch.manual_seed(3)
    recon_only = ttt_loss(
        tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher, weights=LossWeights(ed=0, recon=2, sw=0, cls=0)
    )

    assert recon_only["loss"].item() == pytest.approx(2 * full["recon_loss"].item(), rel=1e-5)


def test_a_prompt_only_window_still_trains_reconstruction(tiny_model, tiny_counts):
    observed, _ = tiny_counts
    # Every position is a prompt cell: no target to match, but masked genes to rebuild.
    torch.manual_seed(0)
    out = ttt_loss(tiny_model, observed, observed, n_kept=observed.shape[1], teacher=None)

    assert out["recon_loss"].item() > 0
    assert out["mmd_loss"].item() == 0.0 and out["cls_loss"].item() == 0.0
    assert math.isfinite(out["loss"].item())


def test_the_objective_trains_the_adapters(tiny_model, tiny_counts):
    observed, truth = tiny_counts
    teacher = teacher_embeddings(tiny_model, truth)
    adapters = inject_lora(tiny_model, stack_targets(tiny_model, "final_block"), rank=2, alpha=4.0)

    ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher)["loss"].backward()

    assert all(a.lora_B.grad is not None and a.lora_B.grad.abs().sum() > 0 for a in adapters)


def test_eval_mode_changes_nothing_in_the_loss_but_dropout(tiny_model, tiny_counts):
    # The validation loss curve is measured in eval mode. With dropout off (the tiny model's
    # dropout is 0), the teacher and every component must equal the training-mode objective.
    observed, truth = tiny_counts
    out = {}
    for training in (True, False):
        tiny_model.train(training)
        torch.manual_seed(7)
        teacher = teacher_embeddings(tiny_model, truth)
        out[training] = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher)

    for name in COMPONENTS + ["loss"]:
        assert out[False][name].item() == out[True][name].item(), name
