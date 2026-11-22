import torch

from stack_ttt.model import load_stack

from conftest import TINY_CONFIG


def test_load_stack_rebuilds_the_saved_model(tiny_model, tiny_counts, tmp_path):
    path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "hyper_parameters": {"model_config": TINY_CONFIG, "n_kept_cell": 6},
            "state_dict": {f"model.{k}": v for k, v in tiny_model.state_dict().items()},
        },
        path,
    )

    model, hparams = load_stack(path)

    assert hparams["n_kept_cell"] == 6
    observed, truth = tiny_counts
    torch.manual_seed(0)
    expected = tiny_model(observed, truth, n_kept_cell=6, return_loss=False, mask_genes=False)["nb_mean"]
    torch.manual_seed(0)
    got = model(observed, truth, n_kept_cell=6, return_loss=False, mask_genes=False)["nb_mean"]
    assert torch.allclose(got, expected)
    assert not model.training
