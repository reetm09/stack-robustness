# Study 4: test-time training (TTT) vs in-context learning (ICL) on STACK

This applies Akyürek et al., *The Surprising Effectiveness of Test-Time Training for
Few-Shot Learning* ([arXiv 2411.07279](https://arxiv.org/abs/2411.07279)), to STACK. For
each held-out instance, fresh low-rank adapters are trained on tasks built from data
available at inference time. The adapted model is then compared with frozen STACK,
which uses prompts alone. The proposal's "Test-Time Training" section is the spec.

Everything for this study lives in `ttt/`, with its own environment. It depends on the
shared scoring package (`../src/stack_robustness`) by path and scores every model with
the same `EvalContext` as the baselines.

```bash
cd ttt
uv sync                                   # torch, arc-stack 0.1.3, and the shared package
uv run pytest                             # CPU only; the GPU is hidden from tests
dibs submit --cpus 4 --mem 8G --vram 6G --time 30m -- uv run python scripts/norman_compare.py --doubles 3
```

## From the paper to STACK

| Akyürek et al. | Here |
|---|---|
| A test task's demonstrations | H(g*): training doubles sharing a gene with the held-out double g* = A+B |
| Leave-one-out tasks | Each neighbour double as a (prompt single, query single, target double) window |
| Invertible augmentations | Cell permutation and subsampling (each window is a fresh draw); prompt/query swap |
| Per-task LoRA | Fresh adapters per held-out double (`adapt.py`), or one shared adapter per context |
| Augmented inference with voting | Per gene, the median over draws within each prompt order, then over the orders |

## Layout (`src/stack_ttt/`)

- `model.py` loads the checkpoint with `torch.load(weights_only=True, mmap=True)`.
- `lora.py` holds the adapters. The default scope is the final block's cross-cell
  attention plus the decoder; `all_cross_cell` is the ablation.
- `objective.py` is STACK's fine-tuning loss, reassembled from the model's own helpers:
  energy distance in gene and frozen-teacher embedding space, masked reconstruction,
  sliced Wasserstein and the f_CLS loss, each weighted. A test checks it equals upstream
  `forward` under the same seed.
- `tasks.py`, `windows.py` and `norman.py` build the tasks and windows. Single
  perturbations' cells are split into disjoint "adapt" and "infer" halves, so adaptation
  never sees the cells a prediction starts from.
- `adapt.py` is the TTT loop: fresh adapters, AdamW, base model frozen.
- `predict.py` runs STACK's own iterative generation (T steps), turns generated cells
  into log-normalized profiles, and does the median-of-medians aggregation.

## STACK facts the code relies on

These were checked against arc-stack 0.1.3 and `Stack-Large-Aligned`:

- 217.8M parameters, 9 layers, 1,600-dimensional cell embeddings, windows of 512 cells.
  Fine-tuning kept 128 prompt cells per window and used learning rate 2e-5.
- Fine-tuning windows hold [prompt cells | query cells]. A random share of query
  positions is revealed as context. Prompt cells cannot attend to query cells.
- Inference windows hold [prompt cells | more prompt cells as context | query cells].
- Generation returns counts sampled from the negative-binomial output. They are
  normalized with the query cells' all-gene totals, the convention of the ground truth.

## Norman neighbourhoods

For the 26 held-out doubles (seed 0), H(g*) has 0 to 15 training doubles; most have 3
to 15. Three are thin: KIF18B+KIF2C has 0, TMSB4X+BAK1 has 1 and SGK1+S1PR2 has 2.
These lean on the null task (control → control) and on reconstruction of their own
prompt cells. JUN+CEBPA has only 54 cells.

## Status

Built and unit-tested: everything above.

Next steps:
1. Frozen STACK (ICL) and TTT scored side by side (`scripts/norman_compare.py`), at T=1 and T=5.
2. TTT with energy distance only at T=1, then the regularizers, f_CLS and T=5.
3. Hyperparameters chosen on validation doubles carved from the training doubles. No
   cell of a held-out double ever enters adaptation or selection.

Deferred:
- The X-Atlas/Orion arm, which needs a slice of the 126 GB Hugging Face copy.
- k-nearest-neighbour retrieval of H(g*).
- The shared context adapter.
- Count-space input augmentations.
