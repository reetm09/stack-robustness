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


def test_revealed_cells_are_both_input_and_reconstruction_target_of_their_slots(tiny_model, tiny_counts):
    # With `revealed`, the slots after the prompt show its cells (for a pair task, more
    # prompt cells, as inference does) and reconstruction asks for those same cells there.
    observed, truth = tiny_counts
    teacher = teacher_embeddings(tiny_model, truth)
    revealed = torch.poisson(torch.full_like(truth, 7.0), generator=torch.Generator().manual_seed(3))
    revealed[:, :N_KEPT] = truth[:, :N_KEPT]
    for seed in range(100):  # a draw that reveals at least two positions
        torch.manual_seed(seed)
        out = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher, revealed=revealed)
        if out["n_context"] >= N_KEPT + 2:
            break
    shown = truth.clone()
    shown[:, N_KEPT:out["n_context"]] = revealed[:, N_KEPT:out["n_context"]]
    torch.manual_seed(seed)
    same = ttt_loss(tiny_model, observed, shown, n_kept=N_KEPT, teacher=teacher)
    torch.manual_seed(seed)
    plain = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher)

    for name in COMPONENTS + ["loss"]:
        assert out[name].item() == pytest.approx(same[name].item(), rel=1e-6), name
    assert out["recon_loss"].item() != pytest.approx(plain["recon_loss"].item(), rel=1e-6)


def test_the_full_loss_is_the_registered_default_and_unknown_kinds_are_refused():
    from stack_ttt.objective import loss_weights

    assert loss_weights("full") == LossWeights()
    with pytest.raises(ValueError):
        loss_weights("energy")


def test_energy_distance_only_keeps_just_the_energy_distance(tiny_model, tiny_counts):
    from stack_ttt.objective import loss_weights

    observed, truth = tiny_counts
    teacher = teacher_embeddings(tiny_model, truth)
    torch.manual_seed(3)
    out = ttt_loss(tiny_model, observed, truth, n_kept=N_KEPT, teacher=teacher, weights=loss_weights("ed"))

    assert out["mmd_loss"].item() > 0 and out["recon_loss"].item() > 0
    assert out["loss"].item() == pytest.approx(out["mmd_loss"].item(), rel=1e-6)


def test_energy_distance_only_gives_a_prompt_only_window_no_gradient(tiny_model, tiny_counts):
    from stack_ttt.objective import loss_weights

    observed, _ = tiny_counts
    adapters = inject_lora(tiny_model, stack_targets(tiny_model, "final_block"), rank=2, alpha=4.0)
    for a in adapters:  # nonzero B, so a gradient would reach A too
        torch.nn.init.normal_(a.lora_B)
    torch.manual_seed(0)
    out = ttt_loss(tiny_model, observed, observed, n_kept=observed.shape[1], teacher=None, weights=loss_weights("ed"))
    out["loss"].backward()

    assert out["loss"].item() == 0.0
    grads = [p.grad for a in adapters for p in (a.lora_A, a.lora_B)]
    assert all(g is not None and g.abs().sum() == 0 for g in grads)
