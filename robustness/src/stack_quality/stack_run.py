"""STACK (frozen Stack-Large-Aligned) on one condition, draw and plate.

For each drug, STACK generates the query cells under the drug prompt and, once per plate,
under the DMSO prompt (same query cells and seed). Per query cell type:

  stack      control reference + mean(generated | drug) - mean(generated | DMSO)
             (Arc's tutorial "synthetic control": cancels shifts STACK applies whatever the prompt)
  stack_raw  control reference + mean(generated | drug) - mean(observed query)

The control reference is the target group's clean DMSO profile, so every model is scored as a
predicted effect added to the same reference. The same corrupted prompts also give each
target's prompt effect (drug minus DMSO T/NK profile), the input of the tabular models and of
source-effect copy.
"""

import time

import numpy as np
from scipy import sparse

from stack_ttt.predict import generate, mean_log_profile

from .conditions import Condition, Inputs
from .corrupt import corrupt
from .pbmc import Cells, library, log_profile


def _corrupted(cells: Cells, rows, cond: Condition, role: str, background, seed) -> sparse.csr_matrix:
    counts = cells.counts[rows]
    if not cond.corrupts(role):
        return counts
    return corrupt(counts, thin_rate=cond.thin_rate, ambient_rate=cond.ambient_rate, background=background,
                   rng=np.random.default_rng(seed))


def run_plate(model, cells: Cells, inputs: Inputs, cond: Condition, draw: int, plate: str, targets, ctrl, panel,
              genelist_path, background, *, num_steps: int = 1, batch_size: int = 8, seed: int = 0,
              control_cache: dict | None = None) -> tuple:
    """Returns ({model: {target id: effect vector on `panel`}}, list of per-call timings)."""
    cols = cells.data_columns(panel)
    gen_seed = 1000 * draw + 1
    p = sorted(set(cells.plate)).index(plate)
    timings = []

    query = _corrupted(cells, inputs.query, cond, "query", background, [seed, draw, p, 1])
    q_lib = library(query)
    q_vocab = query @ cells.to_vocab
    control = _corrupted(cells, inputs.control_prompt, cond, "prompt", background, [seed, draw, p, 2])
    control_profile = log_profile(control, cols)

    def gen(prompt_counts):
        t = time.time()
        out = generate(model, prompt_counts @ cells.to_vocab, q_vocab, cells.vocab, genelist_path,
                       num_steps=num_steps, seed=gen_seed, batch_size=batch_size)
        timings.append(time.time() - t)
        return out

    key = (cond.control_key, draw, plate)
    if control_cache is not None and key in control_cache:
        generated_control = control_cache[key]
    else:
        generated_control = gen(control)
        if control_cache is not None:
            control_cache[key] = generated_control

    by_kind = {k: np.flatnonzero(inputs.query_kind == k) for k in np.unique(inputs.query_kind)}

    def per_kind(gen_counts):
        return {k: mean_log_profile(gen_counts[r], q_lib[r], cells.vocab, panel).to_numpy() for k, r in by_kind.items()}

    gen_ctrl = per_kind(generated_control)
    observed = {k: log_profile(query[r], cols) for k, r in by_kind.items()}

    out = {"stack": {}, "stack_raw": {}, "prompt_effect": {}}
    all_drugs = sorted(set(cells.drug))
    for drug, rows in inputs.prompts.items():
        prompt = _corrupted(cells, rows, cond, "prompt", background, [seed, draw, p, 10 + all_drugs.index(drug)])
        prompt_effect = log_profile(prompt, cols) - control_profile
        gen_drug = per_kind(gen(prompt))
        for t in targets:
            if t.drug != drug or t.plate != plate:
                continue
            c = ctrl.loc[t.group].to_numpy()
            out["stack"][t.id] = c + gen_drug[t.kind] - gen_ctrl[t.kind]
            out["stack_raw"][t.id] = c + gen_drug[t.kind] - observed[t.kind]
            out["prompt_effect"][t.id] = prompt_effect
    return out, timings
