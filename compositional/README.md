# Experiment 2: Combinatorial perturbation prediction

Can STACK combine two separately observed gene perturbations through in-context learning,
without being trained to predict their combined effect? Full plan: [`PLAN.md`](PLAN.md).
Current results: [`EVALUATION.md`](EVALUATION.md).

## In short

Scientists can switch on one gene at a time in a dish of cells and measure what happens, but
testing every *pair* of genes is far too expensive. If a model could predict what a pair does
from the singles alone, it would save a lot of experiments. We test whether STACK can do this:
we show it cells with gene A switched on (the "prompt") and give it cells with gene B switched
on (the "query"), and ask it to predict cells with both. We compare against simply adding the
two single effects together. That simple approach is very strong on this dataset, so the real
test is the handful of gene pairs where the combined effect is *not* just the sum.

## Technical details

### Setup

Everything runs in the shared environment `cs182-finalproject/envs/.venv`:

```bash
source ../../envs/.venv/bin/activate          # from compositional/
uv pip install -e .. -e .                     # shared package + this one (editable)
python -m pytest                              # CPU-only unit tests
```

The shared code lives in `../src/stack_robustness` (split, baselines, metrics, `EvalContext`).
This folder depends on it by path. `../ttt/` (Study 4) is not imported or run.

### Layout

```
compositional/
  PLAN.md               plan, design decisions, step status
  SPLITS.md             which pairs are hidden in each experiment, and why
  EVALUATION.md         auto-generated results tables (do not edit by hand)
  tables/composition_results.tex   auto-generated LaTeX table (\input in the paper)
  src/stack_compositional/
    report.py           scores.csv -> summary tables (Markdown + LaTeX)
    contexts.py         load Norman; build the halves / all-cells scoring contexts
    additivity.py       additive / buffering-synergy / interaction labels (no model)
    splits.py           S-all folds and S-gene held-out-gene splits
  scripts/
    make_table.py       rebuild EVALUATION.md and the LaTeX table
    build_contexts.py   scoring contexts + CPU baselines for one split
    make_additivity_groups.py  -> ../splits/compositional/additivity_groups.csv
    make_splits.py      -> ../splits/compositional/s_all_folds.csv, s_gene_seed*.csv
    make_explore.py     -> explore.ipynb
  tests/                unit tests (synthetic) + leakage checks on the saved split files
```

Coming in later steps (see PLAN.md): `combine.py` (additive, weighted, symmetric, residual,
ridge), `behavior.py` (replacement / preservation / additive / interaction), `stack_run.py`
(STACK generation, GPU) and `scripts/compose_norman.py`.

### Updating the results table

Every run writes `../results/compositional/<split>/scores.csv` in the shared long format
(`panel, model, perturbation, metric, value, gap_closed`). Then:

```bash
python scripts/make_table.py               # test split
python scripts/make_table.py --split val   # validation split
```

Model names it looks for: `additive`, `weighted`, `ridge`, `stack_frozen`,
`stack_finetuned`, `stack_both`. Metrics: `pearson_deltactrl`, `mse`, `wmse`,
`residual_pearson`. Anything missing shows as `--`.

## Status

See the step table in [`PLAN.md`](PLAN.md#steps-one-at-a-time-approval-after-each).
