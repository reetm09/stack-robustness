import pytest
import torch
from torch import nn

from stack_ttt.lora import LoRALinear, adapters_disabled, inject_lora, lora_modules, stack_targets


def _linear():
    torch.manual_seed(0)
    return nn.Linear(3, 2)


def test_a_fresh_adapter_leaves_the_layer_unchanged():
    base = _linear()
    x = torch.randn(5, 3)
    expected = base(x)

    assert torch.allclose(LoRALinear(base, rank=2, alpha=4.0)(x), expected)


def test_adapter_adds_scaled_low_rank_update():
    layer = LoRALinear(_linear(), rank=1, alpha=4.0)  # scaling = alpha / rank = 4
    with torch.no_grad():
        layer.lora_A.copy_(torch.tensor([[1.0, 0.0, 2.0]]))
        layer.lora_B.copy_(torch.tensor([[1.0], [-1.0]]))
    x = torch.tensor([[1.0, 5.0, 1.0]])  # A x = 3

    delta = layer(x) - layer.base(x)

    assert delta[0].tolist() == pytest.approx([12.0, -12.0])


def test_disabled_adapters_give_the_base_output_and_reenable_afterwards():
    layer = LoRALinear(_linear(), rank=1, alpha=1.0)
    with torch.no_grad():
        layer.lora_B.fill_(1.0)
    x = torch.randn(4, 3)
    model = nn.Sequential(layer)

    with adapters_disabled(model):
        assert torch.allclose(model(x), layer.base(x))
    assert not torch.allclose(model(x), layer.base(x))


def test_reset_restores_the_base_output():
    layer = LoRALinear(_linear(), rank=2, alpha=2.0)
    with torch.no_grad():
        layer.lora_B.fill_(0.5)
    x = torch.randn(4, 3)

    layer.reset()

    assert torch.allclose(layer(x), layer.base(x))


def test_stack_targets_cover_the_final_cross_cell_attention_and_the_decoder(tiny_model):
    assert stack_targets(tiny_model, "final_block") == [
        "layers.1.gene_attn.qkv",
        "layers.1.gene_attn.proj",
        "output_mlp.0",
        "output_mlp.3",
    ]
    assert stack_targets(tiny_model, "all_cross_cell")[:4] == [
        "layers.0.gene_attn.qkv",
        "layers.0.gene_attn.proj",
        "layers.1.gene_attn.qkv",
        "layers.1.gene_attn.proj",
    ]


def test_injecting_into_stack_keeps_its_output_and_trains_only_the_adapters(tiny_model, tiny_counts):
    observed, truth = tiny_counts
    torch.manual_seed(0)  # the forward draws a random share of revealed context
    before = tiny_model(observed, truth, n_kept_cell=6, return_loss=False, mask_genes=False)["nb_mean"]

    inject_lora(tiny_model, stack_targets(tiny_model, "final_block"), rank=2, alpha=4.0)
    torch.manual_seed(0)
    after = tiny_model(observed, truth, n_kept_cell=6, return_loss=False, mask_genes=False)["nb_mean"]

    assert torch.allclose(before, after, atol=1e-6)
    trainable = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert trainable == {f"{name}.lora_{ab}" for name in stack_targets(tiny_model, "final_block") for ab in "AB"}
    assert len(lora_modules(tiny_model)) == 4
