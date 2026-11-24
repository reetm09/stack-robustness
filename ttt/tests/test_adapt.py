import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy import sparse

from stack_ttt.adapt import COMPONENTS, AdaptConfig, adapt, task_losses
from stack_ttt.lora import adapters_disabled, inject_lora, lora_modules, stack_targets
from stack_ttt.norman import Cells, single_pools
from stack_ttt.tasks import control_pools, norman_tasks

from conftest import TINY_CONFIG


@pytest.fixture
def setup(tiny_model):
    rng = np.random.default_rng(0)
    condition = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["A+C"] * 20 + ["control"] * 30)
    counts = sparse.csr_matrix(rng.poisson(3.0, size=(len(condition), TINY_CONFIG["n_genes"])).astype(np.float32))
    cells = Cells(counts=counts, library=np.asarray(counts.sum(1)).ravel() + 10, condition=condition,
                  genes=[f"G{i}" for i in range(TINY_CONFIG["n_genes"])])
    pools = single_pools(condition, ["A", "B", "C"], seed=0) | control_pools(condition, seed=0)
    tasks = norman_tasks("A+B", ["A+C"], null=True)
    inject_lora(tiny_model, stack_targets(tiny_model, "final_block"), rank=2, alpha=4.0)
    return tiny_model, cells, pools, tasks


CONFIG = AdaptConfig(steps=4, lr=1e-2, n_cells=16, n_kept=4, seed=0)


def _frozen_state(model):
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" not in k and not k.startswith("cls.")}


def test_adapt_trains_adapters_and_leaves_the_base_model_untouched(setup):
    model, cells, pools, tasks = setup
    before = _frozen_state(model)

    result = adapt(model, cells, tasks, pools, CONFIG)

    after = _frozen_state(model)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert any(layer.lora_B.abs().sum() > 0 for layer in lora_modules(model))
    assert len(result["history"]) == CONFIG.steps
    assert all(math.isfinite(h["loss"]) for h in result["history"])
    assert not model.training


def test_adapt_starts_from_fresh_adapters_and_the_given_classifier(setup):
    model, cells, pools, tasks = setup
    base_cls = {k: v.clone() for k, v in model.cls.state_dict().items()}
    for layer in lora_modules(model):
        torch.nn.init.ones_(layer.lora_B)
    with torch.no_grad():
        for p in model.cls.parameters():
            p.add_(1.0)

    adapt(model, cells, tasks, pools, AdaptConfig(steps=0, n_cells=16, n_kept=4), cls_base=base_cls)

    assert all(layer.lora_B.abs().sum() == 0 for layer in lora_modules(model))
    assert all(torch.equal(v, base_cls[k]) for k, v in model.cls.state_dict().items())


def test_without_classifier_co_adaptation_only_adapters_change(setup):
    model, cells, pools, tasks = setup
    cls_before = {k: v.clone() for k, v in model.cls.state_dict().items()}

    adapt(model, cells, tasks, pools, AdaptConfig(steps=3, lr=1e-2, n_cells=16, n_kept=4, train_cls=False))

    assert all(torch.equal(v, cls_before[k]) for k, v in model.cls.state_dict().items())


def test_adapt_can_start_from_saved_adapters_instead_of_fresh_ones(setup):
    from stack_ttt.lora import adapter_state

    model, cells, pools, tasks = setup
    for layer in lora_modules(model):
        torch.nn.init.ones_(layer.lora_B)
    start = adapter_state(model)
    for layer in lora_modules(model):
        layer.reset()

    adapt(model, cells, tasks, pools, AdaptConfig(steps=0, n_cells=16, n_kept=4), adapter_init=start)

    assert all(torch.equal(layer.lora_B, torch.ones_like(layer.lora_B)) for layer in lora_modules(model))


def test_cosine_schedule_warms_up_linearly_then_decays_to_zero():
    from stack_ttt.adapt import lr_factor

    # torchtune's get_cosine_schedule_with_warmup, which Akyurek et al.'s TTT recipe uses
    assert lr_factor(0, total=10, warmup_steps=2, schedule="cosine") == 0.0
    assert lr_factor(1, total=10, warmup_steps=2, schedule="cosine") == pytest.approx(0.5)
    assert lr_factor(2, total=10, warmup_steps=2, schedule="cosine") == pytest.approx(1.0)
    assert lr_factor(6, total=10, warmup_steps=2, schedule="cosine") == pytest.approx(0.5)
    assert lr_factor(10, total=10, warmup_steps=2, schedule="cosine") == pytest.approx(0.0)
    assert lr_factor(7, total=10, warmup_steps=2, schedule="constant") == 1.0


def test_adapt_follows_the_schedule_and_records_the_learning_rate(setup):
    model, cells, pools, tasks = setup
    config = AdaptConfig(steps=10, lr=1e-2, n_cells=16, n_kept=4, schedule="cosine", warmup=0.2)

    history = adapt(model, cells, tasks, pools, config)["history"]

    assert [h["lr"] for h in history] == pytest.approx([1e-2 * f for f in [0, 0.5, 1, 0.962, 0.854, 0.691, 0.5, 0.309, 0.146, 0.038]], abs=1e-3)


def test_the_monitor_runs_in_eval_mode_before_training_every_n_steps_and_at_the_end(setup):
    model, cells, pools, tasks = setup
    modes = []

    def monitor(m):
        modes.append(m.training)
        return {"x": 1.0}

    result = adapt(model, cells, tasks, pools, AdaptConfig(steps=5, lr=1e-2, n_cells=16, n_kept=4),
                   monitor=monitor, monitor_every=2)

    assert [r["step"] for r in result["monitor"]] == [0, 2, 4, 5]
    assert modes == [False] * 4
    assert len(result["history"]) == 5 and not model.training


def test_the_monitor_leaves_the_training_run_unchanged(setup):
    model, cells, pools, tasks = setup
    cls_base = {k: v.clone() for k, v in model.cls.state_dict().items()}

    def draws(m):  # takes numbers from torch's global generator, as the masking in the loss does
        torch.rand(100)
        return {}

    torch.manual_seed(7)  # fresh adapters start from a random A
    plain = adapt(model, cells, tasks, pools, CONFIG, cls_base=cls_base)["history"]
    torch.manual_seed(7)
    watched = adapt(model, cells, tasks, pools, CONFIG, cls_base=cls_base, monitor=draws, monitor_every=1)["history"]

    assert [h["loss"] for h in watched] == [h["loss"] for h in plain]


def test_the_first_measurement_is_of_the_starting_model(setup):
    model, cells, pools, tasks = setup
    cls_base = {k: v.clone() for k, v in model.cls.state_dict().items()}
    with adapters_disabled(model):  # fresh adapters (B = 0) leave the base model unchanged
        expected = task_losses(model, cells, tasks, pools, CONFIG)

    result = adapt(model, cells, tasks, pools, CONFIG, cls_base=cls_base,
                   monitor=lambda m: task_losses(m, cells, tasks, pools, CONFIG), monitor_every=2)

    first, last = result["monitor"][0], result["monitor"][-1]
    assert {k: first[k] for k in expected} == pytest.approx(expected, rel=1e-6)
    assert last["step"] == CONFIG.steps and last["loss"] != pytest.approx(first["loss"], rel=1e-6)


def test_task_losses_repeat_exactly_and_leave_the_global_generator_alone(setup):
    model, cells, pools, tasks = setup
    state = torch.get_rng_state()

    first = task_losses(model, cells, tasks, pools, CONFIG)

    assert torch.equal(torch.get_rng_state(), state)
    assert task_losses(model, cells, tasks, pools, CONFIG) == first
    assert set(first) == set(COMPONENTS) and all(math.isfinite(v) for v in first.values())


def test_fresh_adapters_start_from_the_runs_seed_whatever_ran_before(setup):
    # A resumed run skips finished pairs, so the work done before a pair's adaptation differs
    # from an uninterrupted run's. A pair's fresh adapters must depend on the seed alone.
    model, cells, pools, tasks = setup
    cls_base = {k: v.clone() for k, v in model.cls.state_dict().items()}
    runs = []
    for earlier in (1, 2):
        torch.manual_seed(earlier)  # stands in for whatever drew random numbers before
        runs.append([h["loss"] for h in adapt(model, cells, tasks, pools, CONFIG, cls_base=cls_base)["history"]])

    assert runs[0] == runs[1]


def test_adapting_with_revealed_prompt_cells_trains_and_differs_from_revealed_targets(setup):
    model, cells, pools, tasks = setup
    cls_base = {k: v.clone() for k, v in model.cls.state_dict().items()}
    runs = {}
    for reveal in ("target", "prompt"):
        config = replace(CONFIG, steps=6, reveal=reveal)
        runs[reveal] = [h["loss"] for h in adapt(model, cells, tasks, pools, config, cls_base=cls_base)["history"]]

    assert all(math.isfinite(x) for x in runs["prompt"])
    assert runs["prompt"] != runs["target"]
