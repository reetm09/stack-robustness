# Study 1: target-cell-line fine-tuning of STACK

Fine-tunes Stack-Large-Aligned on a target cell line's training knockdowns, and saves it in a
form the evaluation code can load beside zero-shot STACK. The proposal's "New Cell Line
Prediction" section is the spec. This directory adds no evaluation and changes no shared
code. It reuses `stack_ttt` (Study 4) and `stack_robustness` (shared scoring) unchanged.

## Status

- **Loads X-Atlas/Orion and runs from the command line** (`xatlas.py`, `run.py`; see
  "First run" below). The loader follows the release's schema as read from its files on
  2026-11-24. It is tested only on synthetic files with that schema.
- **Depends on code not yet on `main`.** `stack_ttt` (`ttt/`) and the shared
  `stack_robustness` package are on Andrew's branch. `pyproject.toml` points at them by path
  (`..` and `../ttt`), so `uv sync` here works only once they are on the same branch.
- **Tested** with the tests in `tests/`, on synthetic cells and a tiny STACK, on CPU, with the
  server's package versions (torch 2.14.1, arc-stack 0.1.3, pytorch-lightning 2.6.6, scvi-tools
  1.5.2, geomloss 0.3.1, pandas 3.1.1, anndata 0.13.4, numpy 2.5.5, scipy 1.18.1). **Not yet
  run** on real data, on the real checkpoint, or on a GPU.
- **Needs Andrew's branch from `c0ffb27` on**, where `task_window` returns three values.
  Validation windows follow `AdaptConfig.reveal`, like training. The library default is
  `"target"` (reveal true target cells); Andrew's Study 4 scripts now use `"prompt"`, which
  matches STACK's inference. Which one Study 1 uses is a team decision.

## Pending team confirmation

1. **Base branch.** Andrew's shared package and `ttt/` need to reach `main`.
2. **Dataset and schema.** Answered by Soodabeh's run (see "The X-Atlas/Orion split" at the
   end). One open point: she reports that `sample` is the GEM batch (one file per batch).
3. **Split.** Built on 2026-11-23 from the real data, in `splits/xatlas_orion/`. Its settings
   still need the team's agreement: 70/15/15, seed 0, one gene split for both directions. Every
   released cell already passes the guide filter, so no extra filter was applied.
4. **Remaining method choices.**
   - Loss: STACK's full fine-tuning loss (the default here, as in `ttt/`) or energy distance
     only (the proposal's Eq. 2; `LossWeights(recon=0, sw=0, cls=0)`).
   - Selection: by validation loss, or by a scored validation metric, which needs an
     X-Atlas validation `EvalContext`.
   - Step count and learning rate. The defaults are `norman_ttt.py`'s shared-adapter
     settings, not tuned for any cell line.
   - What windows reveal after the prompt (`--reveal target` or `prompt`), and whether f_CLS
     is trained (`--freeze-cls`). Andrew's Study 4 version 2 reveals prompt cells.
   - How many genes and cells a full run uses. All training cells would need about 130 GB of
     memory, so runs use a subset (`--train-genes`, `--max-cells-per-gene` and so on).

## The split

`src/stack_cellline/split.py` applies the proposal's rule (PDF p. 3; `proposal/main.tex:106`) to
the real cell metadata and writes the split once, for everyone to use:

- **Eligible targets:** at least 100 cells in **each** line, counted only in GEM batches that hold
  that line's non-targeting controls.
- **Genes:** eligible targets are split into train, validation and test **by gene**, so all of a
  gene's guides land together. One gene split serves both directions (HEK293T → HCT116 and the
  reverse), so a test gene is never a training gene in either direction.
- **Controls:** each line's non-targeting control cells are split into train, validation and test
  pools **within each GEM batch**.

Settings on this branch, still to be agreed by the team: 70/15/15 for both genes and controls,
and seed 0. Reruns with the same inputs give identical files.

It reads only cell metadata (`obs` of an `.h5ad`, or a `.csv`/`.tsv`/`.parquet` table), never
expression data, and needs only pandas and anndata. Run it from the repo root. Replace the
`<...>` parts with the dataset's real file and column names:

```bash
PYTHONPATH=cellline/src python -m stack_cellline.split \
    --obs <file.h5ad> --line-col <cell line column> --gene-col <target gene column> \
    --batch-col <GEM batch column> --guide-col <guide column> \
    --control-label <non-targeting label> --lines HEK293T HCT116 \
    --out cellline/splits/xatlas_orion
```

- If each cell line is in its own file with no cell-line column, drop `--line-col` and give
  `--obs HEK293T=<file> HCT116=<file>`.
- To count only guide-filtered cells, add `--query "<pandas condition>"`.
- It refuses to overwrite an existing split unless given `--overwrite`.

Outputs in `--out`:

- `genes.csv`: one row per target gene, with its split (`train`, `val`, `test` or `excluded`), its
  cells per line in batches with controls, its number of guides, and why it was excluded.
- `controls.csv.gz`: one row per pooled control cell: `cell_id`, `line`, `batch`, `pool`.
- `split.json`: the settings, input files, column names and counts.

## First run

The proposal's first step (PDF p. 4) is to fine-tune on a small training and validation
subset and measure memory and runtime before scaling. `stack_cellline.run` does that in two
steps.

**Inputs:**

- the checkpoint and STACK's vocabulary file (paths as in Andrew's README:
  `data/raw/stack/bc_large_aligned.ckpt`, `data/raw/stack_vocab/basecount_1000per_15000max.pkl`);
- `gene_metadata.parquet` (0.9 MB), from the release's `metadata/` folder;
- the per-line metadata tables written by `scripts/fetch_xatlas_obs.py`;
- the split in `splits/xatlas_orion/`;
- the batch files: either a local copy of the release's `data/` folder, or
  `hf://datasets/Xaira-Therapeutics/X-Atlas-Orion/data`, which streams only the files a run
  needs (0.3–0.5 GB each).

**Code:** until Andrew's code reaches `main`, put his branch next to this one and add it to
`PYTHONPATH`. The run also imports scanpy and scikit-learn through his shared package.

```bash
git worktree add ../andrew-code origin/andrew
export PYTHONPATH=cellline/src:../andrew-code/src:../andrew-code/ttt/src
```

**Step 1: plan.** This reads no expression data. It writes `plan.json` (genes, cells and batch
files) and prints a summary.

```bash
python -m stack_cellline.run --plan-only \
    --checkpoint data/raw/stack/bc_large_aligned.ckpt \
    --vocab data/raw/stack_vocab/basecount_1000per_15000max.pkl \
    --gene-metadata <dir>/gene_metadata.parquet \
    --obs HEK293T=<obs_dir>/HEK293T_obs.parquet HCT116=<obs_dir>/HCT116_obs.parquet \
    --split cellline/splits/xatlas_orion \
    --data hf://datasets/Xaira-Therapeutics/X-Atlas-Orion/data \
    --train-genes 50 --val-genes 10 --max-cells-per-gene 256 --max-controls 2000 \
    --batches-per-line 10 \
    --out results/cellline/first_run
```

**Step 2: short run.** Use the same command without `--plan-only` and add `--steps 50`. It
writes `adapter.pt`, `history.jsonl`, `val.json` and `run.json` next to `plan.json`.

The caps above are a starting point, not a recommendation. Ten batch files per line give
roughly 10–20 cells per gene, which is enough to check the pipeline, memory and runtime, not
to judge accuracy.

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

## The X-Atlas/Orion split (run 2026-11-23)

`cellline/splits/xatlas_orion/` is the Study 1 split, built once from the real data with the
default settings (70/15/15, seed 0, at least 100 cells per line).

**Data source.** Hugging Face `Xaira-Therapeutics/X-Atlas-Orion`: 332 parquet files, one per
GEM batch (`data/<LINE>_Batch<N>.parquet`), about 130 GB in total. There is no cell-line
column; the line comes from the file name. `scripts/fetch_xatlas_obs.py` reads only the
per-cell metadata columns (not the expression columns) and writes one table per line:

```bash
python cellline/scripts/fetch_xatlas_obs.py <out_dir>   # needs pandas, pyarrow, huggingface_hub
```

**Split command.**

```bash
PYTHONPATH=cellline/src python -m stack_cellline.split \
    --obs HEK293T=<out_dir>/HEK293T_obs.parquet HCT116=<out_dir>/HCT116_obs.parquet \
    --cell-id-col cell_barcode --gene-col gene_target --batch-col sample \
    --guide-col guide_target --control-label Non-Targeting \
    --lines HEK293T HCT116 --out cellline/splits/xatlas_orion
```

**Result** (`splits/split_summary.txt`):

```
18319 targets: 8805 train, 1887 val, 1887 test, 5740 excluded
HEK293T controls in 223 batches: 153188 train, 32825 val, 32825 test
HCT116 controls in 109 batches: 116045 train, 24866 val, 24866 test
```

**Notes on the data** (column listing in `splits/xatlas_orion_columns.txt`):

- 4,534,299 HEK293T cells and 3,409,169 HCT116 cells. Every GEM batch has non-targeting controls.
- Every released cell already passes the guide filter (`pass_guide_filter == 1`, `num_features == 2`),
  so no `--query` was used.
- `guide_target` holds a guide pair as one value (e.g. `ST14_P1P2-1|ST14_P1P2-2`), so
  `n_guides` in `genes.csv` counts guide pairs and is usually 1.
- Excluded genes: 3,466 have fewer than 100 cells in both lines, 2,141 only in HCT116, 133 only in HEK293T.
- Expression is stored sparsely per cell as `gene_token_id` and `gene_expression` (raw counts);
  `metadata/gene_metadata.parquet` maps token IDs to Ensembl IDs and gene symbols.
