"""End to end on synthetic cells and a tiny STACK: train, log, save, reload."""

import json
import math

import pytest
import torch
from stack_ttt.adapt import COMPONENTS, AdaptConfig
from stack_ttt.lora import lora_modules

from stack_cellline.artifact import load_finetuned, read_settings
from stack_cellline.finetune import FinetuneConfig, finetune, validation_loss
from stack_cellline.tasks import validation_rows

from conftest import VAL_GENES

CONFIG = FinetuneConfig(
    rank=2, alpha=4.0, adapt=AdaptConfig(steps=4, lr=1e-2, n_cells=16, n_kept=4, seed=0), val_windows=2
)


@pytest.fixture
def run(tiny_checkpoint, tiny_data, tmp_path):
    data, split = tiny_data
    return finetune(tiny_checkpoint, data, split, tmp_path / "run", CONFIG)


def assert_same_loss(got, expected):
    assert {k: got[k] for k in COMPONENTS} == pytest.approx({k: expected[k] for k in COMPONENTS})
    for gene in expected["per_gene"]:
        assert got["per_gene"][gene] == pytest.approx(expected["per_gene"][gene])


def _base_state(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" not in k and not k.startswith("cls.")}


def test_run_logs_training_and_validation_loss(run, tmp_path):
    out = tmp_path / "run"
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    val = json.loads((out / "val.json").read_text())
    summary = json.loads((out / "run.json").read_text())

    assert len(history) == CONFIG.adapt.steps
    assert all(math.isfinite(h["loss"]) for h in history)
    for stage in ("before", "after"):
        assert math.isfinite(val[stage]["loss"])
        assert sorted(val[stage]["per_gene"]) == sorted(VAL_GENES)
    assert summary["val_loss_after"] == val["after"]["loss"]
    trained_on = {h["target"].split("|")[1] for h in history}
    assert trained_on <= set(summary["settings"]["train_genes"])
    assert not trained_on & set(VAL_GENES)


def test_only_adapters_and_classifier_change(run, tiny_model):
    # inject_lora keeps each wrapped layer's frozen weights under "<layer>.base.".
    released = tiny_model.state_dict()
    base = {k.replace(".base.", "."): v for k, v in _base_state(run.model).items()}
    assert base.keys() == {k for k in released if not k.startswith("cls.")}
    assert all(torch.equal(v, released[k]) for k, v in base.items())
    assert any(layer.lora_B.abs().sum() > 0 for layer in lora_modules(run.model))
    assert any(not torch.equal(v, tiny_model.cls.state_dict()[k]) for k, v in run.model.cls.state_dict().items())


def test_saved_adapter_reloads_to_the_trained_model(run, tiny_checkpoint, tiny_data):
    reloaded = load_finetuned(tiny_checkpoint, run.adapter_path)

    trained, again = run.model.state_dict(), reloaded.state_dict()
    assert trained.keys() == again.keys()
    assert all(torch.equal(trained[k], again[k]) for k in trained)
    assert not reloaded.training
    settings = read_settings(run.adapter_path)
    assert settings["targets"][-2:] == ["output_mlp.0", "output_mlp.3"]

    data, split = tiny_data
    rows = validation_rows(data, split)
    assert_same_loss(validation_loss(reloaded, data.counts, rows, CONFIG), run.val_after)


def test_zero_shot_is_the_released_checkpoint_and_matches_the_before_loss(run, tiny_checkpoint, tiny_model, tiny_data):
    zero_shot = load_finetuned(tiny_checkpoint)

    assert not lora_modules(zero_shot)
    assert not any(p.requires_grad for p in zero_shot.parameters())
    assert all(torch.equal(v, tiny_model.state_dict()[k]) for k, v in zero_shot.state_dict().items())
    data, split = tiny_data
    rows = validation_rows(data, split)
    assert_same_loss(validation_loss(zero_shot, data.counts, rows, CONFIG), run.val_before)


def test_validation_follows_the_reveal_setting(tiny_model, tiny_data):
    data, split = tiny_data
    rows = validation_rows(data, split)
    shows_prompt = FinetuneConfig(rank=2, alpha=4.0, val_windows=2,
                                  adapt=AdaptConfig(steps=4, lr=1e-2, n_cells=16, n_kept=4, seed=0, reveal="prompt"))

    target = validation_loss(tiny_model, data.counts, rows, CONFIG)
    prompt = validation_loss(tiny_model, data.counts, rows, shows_prompt)

    assert math.isfinite(prompt["loss"]) and prompt["loss"] != target["loss"]
    assert prompt == validation_loss(tiny_model, data.counts, rows, shows_prompt)


def test_validation_is_repeatable_and_leaves_the_random_state_alone(tiny_model, tiny_data):
    data, split = tiny_data
    rows = validation_rows(data, split)
    state = torch.get_rng_state()

    first = validation_loss(tiny_model, data.counts, rows, CONFIG)
    second = validation_loss(tiny_model, data.counts, rows, CONFIG)

    assert first == second
    assert torch.equal(state, torch.get_rng_state())
    assert all(p.requires_grad for p in tiny_model.parameters())  # flags restored
