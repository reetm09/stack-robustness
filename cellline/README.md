# Study 1: target-cell-line fine-tuning of STACK

Fine-tunes Stack-Large-Aligned on a target cell line's training knockdowns, and saves it in a
form the evaluation code can load beside zero-shot STACK. The proposal's "New Cell Line
Prediction" section is the spec. This directory adds no evaluation and changes no shared
code. It reuses `stack_ttt` (Study 4) and `stack_robustness` (shared scoring) unchanged.

## Status

- **Python API only.** No file loader and no command line yet. They wait on the decisions below.
- **Depends on code not yet on `main`.** `stack_ttt` (`ttt/`) and the shared
  `stack_robustness` package are on Andrew's branch. `pyproject.toml` points at them by path
  (`..` and `../ttt`), so `uv sync` here works only once they are on the same branch.
- **Tested** with the unit and smoke tests in `tests/`, on synthetic cells and a tiny STACK,
  on CPU. **Not yet run** on real data, on the real checkpoint, or on a GPU.

## Pending team confirmation

1. **Base branch.** Andrew's shared package and `ttt/` need to reach `main`.
2. **Dataset and schema.** Where the X-Atlas/Orion slice is and its format. The `obs` columns
   for cell line, target gene, guide, GEM batch and the non-targeting label. Where the raw
   counts and library sizes are.
3. **Split.** Who builds it and its file format: training, validation and test genes, the
   per-GEM-batch control pools, and the eligibility rule (≥100 cells per line per target).
4. **Remaining method choices.**
   - Loss: STACK's full fine-tuning loss (the default here, as in `ttt/`) or energy distance
     only (the proposal's Eq. 2; `LossWeights(recon=0, sw=0, cls=0)`).
   - Selection: by validation loss, or by a scored validation metric, which needs an
     X-Atlas validation `EvalContext`.
   - Step count and learning rate. The defaults are `norman_ttt.py`'s shared-adapter
     settings, not tuned for any cell line.

## The task

For each knockdown g, a window holds:

- **prompt:** g's cells measured in the source line;
- **query:** non-targeting controls of the target line;
- **target:** g's cells measured in the target line.

This is the layout the proposal gives the evaluation. Training uses the training genes with
the training control pool. Validation uses the validation genes with the validation pool.

`check_split` refuses data in which any of these happen:

- a test gene's cells are loaded;
- a target-line control cell is in neither pool (test-pool controls must not be loaded);
- the gene sets or the control pools overlap;
- a gene lacks cells in one line.

## What is trained

`stack_ttt.lora` scope `final_block`:

- LoRA (rank 8, alpha 16) on the last layer's cross-cell attention (`gene_attn.qkv`, `gene_attn.proj`);
- LoRA on the decoder (`output_mlp.0`, `output_mlp.3`);
- the unmasking classifier f_CLS, in full.

Everything else is frozen. That includes the rest of the final block: its within-cell
attention, MLP and LayerNorms. The loss is STACK's fine-tuning loss (`stack_ttt.objective`),
with the base model (adapters off) as a fixed teacher. Upstream arc-stack instead trains every
parameter against a moving-average teacher.

## Use

```python
from stack_cellline.finetune import FinetuneConfig, finetune
from stack_cellline.tasks import CellLineData, Split

data = CellLineData(counts=..., genes=vocab, line=..., target=...)  # cells on STACK's vocabulary
split = Split(source_line="HEK293T", target_line="HCT116", control_label=...,
              train_genes=[...], val_genes=[...], test_genes=[...],
              train_controls=rows, val_controls=rows)
result = finetune("data/raw/stack/bc_large_aligned.ckpt", data, split, "results/cellline/run0",
                  FinetuneConfig(), device="cuda")
```

It writes these files to the output directory:

- `adapter.pt`: the adapters and f_CLS, in the shared adapter's `{"adapters", "cls", "settings"}`
  layout. The settings record the adapted layers, rank, alpha, training settings and genes.
- `history.jsonl`: per training step, the task, learning rate and each loss component.
- `val.json`: validation loss before training (zero-shot) and after, per gene and mean.
- `run.json`: settings, cost (seconds, peak GPU memory) and the mean validation losses.

`adapt()` has no per-step hook, so validation runs only before and after training. Compare
settings across runs by validation loss, as `ttt/PROTOCOL.md` does. A small optional
`on_step` callback in `stack_ttt.adapt.adapt` would allow validation during a run.

## Loading for evaluation

```python
from stack_cellline.artifact import load_finetuned

zero_shot = load_finetuned(checkpoint)                       # the released weights
fine_tuned = load_finetuned(checkpoint, "results/cellline/run0/adapter.pt")
```

Both come back in eval mode with every parameter frozen. Feed both through the same
generation (`stack_ttt.predict.generate` and `mean_log_profile`) with identical prompts and
queries, and score them with the same `EvalContext`.

Two cautions:

- Do not build zero-shot by wrapping the fine-tuned model in `adapters_disabled`. That keeps
  the fine-tuned f_CLS.
- Keep both arms frozen. On CPU, whether parameters require gradients shifts STACK's
  embeddings by about 1e-6 even under `torch.no_grad`. Energy distance amplifies that to
  about 1e-4 relative.

## Tests

```bash
uv run pytest   # once ../ttt and the shared package are on this branch
```
