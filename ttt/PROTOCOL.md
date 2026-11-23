# Study 4 on Norman: protocol, fixed before any validation or test result

Written 22 November 2026, after the 3-double pilot (`results/ttt/norman_pilot3`) and before
any run of `scripts/norman_ttt.py`. Changes after this point are listed at the end, with
reasons.

## Question

Does test-time training (TTT) improve STACK's prediction of Norman double perturbations
over three alternatives?
- Frozen STACK (ICL).
- One adapter shared by every pair and trained on all training doubles.
- The same TTT run on random unrelated pairs.

## Splits

- **Test.** The 26 held-out doubles of `splits/norman2019_conditions.csv` (seed 0). They are
  scored once, with the settings frozen below.
- **Validation.** 21 doubles carved from the 105 training doubles by `scripts/norman_baselines.py
  --validation` (seed 0). That run drops the test doubles' cells entirely. All tuning uses
  validation only. For a validation run, the training doubles are the other 84.
- **Single-perturbation cells.** Split into an adapt half and an infer half (seed 0).
  Training windows draw on the adapt half and predictions start from the infer half. The
  held-out pair's own infer cells enter TTT only through masked reconstruction.

## Arms

The six arms are {frozen, shared} × {no TTT, TTT on neighbours, TTT on random pairs}:

| | no TTT | TTT on neighbours | TTT on random pairs |
|---|---|---|---|
| frozen STACK | `icl` | `ttt` | `ttt_random` |
| shared adapter | `shared` | `shared_ttt` | `shared_ttt_random` |

- **Neighbours** are the training doubles that share a gene with the held-out pair, in
  both prompt orders.
- **Random pairs** are as many training doubles sharing no gene with it, drawn with a seed.
  Both TTT arms add the null task and masked reconstruction of the pair's own single cells.
  They use the same number of tasks, steps and seeds.
- **Starting points.** The shared adapter is the task-format fine-tuning that Akyürek et
  al.'s TTT starts from (`marc/README.md`, `base_checkpoint_dir` "finetuned path"). TTT
  restores it after every pair.
- **References with no model:** the query cells (both prompt orders, and one order), the
  additive baseline, the technical and interpolated duplicates, and the mean and control
  baselines.

## Scoring

- **Headline: diff.** The prediction is the observed query profile, plus STACK's output
  with the prompt, minus its output with a control prompt, with seeds paired. Arc's own
  tutorial (`notebooks/tutorial-predict.ipynb` in ArcInstitute/stack) makes the same
  correction: "To remove artifacts from differential expression predictions, we apply a
  synthetic control approach using control T cells as the reference." Raw scores are
  reported alongside.
- **Primary metric: wMSE.** This is Miller et al.'s DE-weighted mean squared error; lower is
  better. It is computed on the STACK gene panel (11,789 genes), diff-scored, with both
  prompt orders combined by median of medians. It was chosen because it is sensitive to
  scale and is the best calibrated of our metrics on this panel: the interpolated duplicate
  closes 0.86 of its gap, against 0.42 for Pearson Δctrl.
- **Secondary:** Pearson Δctrl, Pearson Δpert and MSE on both panels, and T = 5 against T = 1.

## Tuning on validation (T = 1, 2 draws)

**Fixed settings.** These follow Akyürek et al.'s code (`marc/configs/ttt/8B_lora_single_device.yaml`
and `marc/test_time_train.py`):
- AdamW with weight decay 0.01;
- a cosine schedule with linear warmup over the first 10% of steps;
- LoRA alpha 16 and rank 8;
- f_CLS trained alongside;
- the null task on.

**Steps:**
1. **Shared adapter.** Try scope {final block + decoder, cross-cell attention in all layers +
   decoder} × learning rate {1e-4, 3e-4}, each for 2,000 steps. Select the setting with the
   lowest median validation wMSE of `shared_T1_diff`.
2. **TTT learning rate.** With the selected scope, try {3e-5, 1e-4} at 50 steps, separately
   for `shared_ttt` (from the selected shared adapter) and for `ttt` (from frozen STACK).
3. **TTT steps.** For each of those two arms, run 250 steps at its better learning rate.
   Each arm keeps whichever of its three settings has the lowest median validation wMSE.
   The random-pair arms copy their neighbour arm's settings.

**Ties.** A setting within 1% of the best goes to the cheaper option: fewer steps, then
the smaller scope.

## Test run

One run of the six arms on the 26 test doubles, with the selected settings. Every arm is
generated at T = 1; `icl`, `shared`, `ttt` and `shared_ttt` are also generated at T = 5.
Each prediction uses 2 draws.

## Reporting

- **Medians and paired differences.** Report the median over pairs for every arm and every
  reference. For each comparison below, give the paired per-pair difference in wMSE, with a
  bootstrap 95% interval (10,000 resamples of pairs) and the number of pairs that improved.
  - `shared_ttt − shared`, `shared_ttt − shared_ttt_random`;
  - `ttt − icl`, `ttt − ttt_random`;
  - `shared − icl`;
  - every STACK arm minus the both-orders query cells, and minus the additive baseline.
- **Pair subsets.** Report each comparison on three sets:
  - all 26 pairs;
  - the 23 pairs left after removing the pilot pairs (AHR+FEV, AHR+KLF1 and BPGM+SAMD1);
  - the 9 non-additive pairs. These are fixed now as the largest additive residuals by wMSE
    in `results/norman_baselines`: DUSP9+SNAI1, DUSP9+PRTG, JUN+CEBPA, AHR+FEV, KLF1+COL2A1,
    CBL+PTPN12, BPGM+SAMD1, MAP2K6+SPI1 and AHR+KLF1. All three pilot pairs are in it.
- **Gain against the linear baseline's error.** This is the spec's test: the Spearman
  correlation across pairs between the TTT gain in wMSE (`shared` minus `shared_ttt`, and
  `icl` minus `ttt`) and the additive baseline's wMSE.
- **Cost:** seconds per pair and arm, and peak GPU memory.

## What does not port from Akyürek et al.'s code, and what replaces it

- **Leave-one-out over a task's own demonstrations** (`marc/ttt/preprocess.py:120-133`).
  A held-out pair has no demonstrations of its own, so its neighbouring training pairs
  take their place, one practice window each.
- **Loss on every in-context demonstration's output** (their torchtune fork). A STACK
  window holds one prompt condition, not a sequence of demonstrations.
- **Geometric and colour augmentations applied to whole tasks** (`preprocess.py:135-142`).
  Each window resamples and reshuffles cells instead, and the prompt–query swap leaves the
  joint target unchanged.
- **Up to 250 augmented tasks per test task, 2 epochs at batch size 2**
  (`preprocess.py:196`, `marc/README.md`). Here the budget is a number of steps over
  freshly drawn windows, and it is tuned.
- **Voting within each transformation, then across transformations**
  (`marc/arclib/voting.py`). Here, per gene, take the median over draws within each
  prompt order, then over the orders.
- **LoRA settings.** Theirs is rank 64 to 128 on q, v and the MLP of every layer. The spec
  puts LoRA on STACK's decoder and final block. Its ablation, cross-cell attention in every
  layer, is the nearest STACK analogue and is in the tuning grid.

## Changes after this point

None yet.
