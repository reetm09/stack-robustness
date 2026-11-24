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
- f_CLS frozen (version 1 trained it alongside; see Changes);
- the null task on.

**Steps:**
1. **Shared adapter.** Try scope {final block + decoder, cross-cell attention in all layers +
   decoder} × learning rate {1e-4, 3e-4}, each for 2,000 steps. Select the setting with the
   lowest median validation wMSE of `shared_T1_diff`.
2. **TTT learning rate.** With the selected scope, try {3e-5, 1e-4, 3e-4} at 50 steps, separately
   for `shared_ttt` (from the selected shared adapter) and for `ttt` (from frozen STACK).
3. **TTT steps.** For each of those two arms, run 250 steps at its better learning rate.
   Each arm keeps whichever of its four settings has the lowest median validation wMSE.
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
- **Prompt use (exploratory, added with version 2 before any test result).** For each arm
  and each pair in both prompt orders, take the shift that replacing the control prompt with
  A's cells causes. Report its correlation with A's own effect on A's top 200 genes, and its
  size relative to that effect (`stack_ttt.diagnostics.prompt_use`; README, "What frozen
  STACK does with a Norman prompt").

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

The first two changes were made before any validation score. The third was made after
stage 1's scores but before any TTT result, and changes no setting or rule. The fourth
(version 2) was made after version 1's stage 1 and its 50-step TTT scores. The test set is
still untouched, and no selection rule changed.

- **2026-11-22: a workaround for small query sets.** arc-stack 0.1.3 fills a short
  window with at most one extra copy of the query cells. A query set smaller than half
  a window's query slots therefore raised IndexError; one validation pair has 81 infer
  cells. `predict.generate` now repeats such a set until it fills a window and keeps the
  first copy's generated cells. Larger query sets run exactly as before.
- **2026-11-23: a validation loss curve for the shared adapter.** The course report asks
  for training and validation loss curves (as the `182-clear-research-paper` skill reads
  the course rules). In the validation run, training the shared adapter now records its
  loss every 200 steps, in eval mode, on fixed windows:
  - one window per task of the 21 validation doubles, in both orders;
  - the same for 21 training doubles chosen with the run's seed, as a comparable
    training curve.

  The training run is the same with or without the curve, because the curve's random
  draws are kept apart from training's (tested).

  The validation doubles' cells are loaded for this curve only. No pool or training task
  may use them (asserted), and the test run still never loads the test doubles.

  The curve is a diagnostic, not a selection or early-stopping rule. The 2,000 steps and
  the wMSE selection rule stand. The first setting (final block, 1e-4) is rerun from
  scratch so that all four settings have the curve; its earlier adapter and 20 pairs
  were set aside unscored.
- **2026-11-23: fresh adapters start from the run's seed.**
  - **The problem:** `adapt()` re-initialised LoRA A from torch's global generator before
    seeding it. A `ttt` or `ttt_random` pair's starting adapter therefore depended on
    whatever ran before it in the process, so a resumed run started differently from an
    uninterrupted one.
  - **The fix:** it now seeds first (tested).
  - **Scope:** stage 1's shared adapters were trained before this change. Every TTT run,
    and the test run's shared adapter, use the new initialisation.

- **2026-11-23: version 2, after reading STACK's paper and code.** All validation is redone
  in `results/ttt/val_v2`. Version 1 (`results/ttt/val`) stays as the comparison in which
  true targets were revealed.
  - **Revealed slots hold prompt cells.**
    - What changed: adaptation windows now reveal more cells of the prompt's condition
      after the prompt (`--reveal prompt`), drawn with the prompt cells as one sample. That
      is A's cells for a pair task and control cells for the null task, and those cells are
      the reconstruction targets of their slots (tested).
    - Why: STACK's post-training reveals true cells there, which share the prompt's
      condition, and its inference fills those slots with prompt cells. Version 1 revealed
      true A+B cells, which inference never shows, so an adapter could read the target from
      them.
  - **f_CLS is frozen.**
    - At T = 1 it plays no part, because every cell is replaced in one step.
    - Trained on these windows, it would learn to tell A's cells from the query positions'
      outputs. For T > 1, STACK's generation keeps first the cells f_CLS scores as most
      like the prompt (read from the code, not measured).
  - **Renamed genes are mapped in.** 713 Norman genes that HGNC has renamed are moved into
    the vocabulary slots they were missing from (`stack_robustness.genes.vocabulary_aliases`).
    Without this, STACK never sees 2.4% of Norman's counts. The scoring genes and the truth
    are unchanged.
  - **The TTT learning rates add 3e-4,** marc's config value. In version 1, the best rate
    sat at the top of its grid both in stage 1 (both scopes) and for `ttt` at 50 steps, so
    a null result could have been under-training.
  - **Unchanged, and why:**
    - Prompt size: prompts of 333 or more cells and smaller ones gave the same prompt
      statistic.
    - The T = 1 headline: this protocol fixes it.
    - The frozen teacher: Arc's README calls its procedure frozen-teacher fine-tuning, and
      2,000 steps would see at most 4 teacher updates.
    - No confidence filtering: the paper states one only for Perturb Sapiens.
    - Adapters instead of full fine-tuning: that is the TTT recipe.

## Decisions taken by these rules

- **Version 2's first job (2026-11-23).**
  - **Scores:** the shared adapter (final block, 3e-4) scored 0.0177 under the
    inference-consistent layout, against 0.0163 in version 1. Its prompt correlation fell
    from +0.67 to +0.58, and its shift from 31% to 24% of A's effect. Frozen STACK with the
    renamed genes scored 0.0250, against 0.0260.
  - **Confound:** the layout and the renaming are confounded here. A validation-only run
    (`results/ttt/val_v2_target`: target reveal, renaming on) separates them.
    - *Correction (2026-11-24):* version 2 also froze f_CLS, so three things changed from
      version 1, not two. The target-reveal run keeps version 2's renaming and frozen
      f_CLS, so it isolates the layout only. The renaming and f_CLS stay confounded with
      each other.
  - **Decision:** prompt reveal stays primary, as registered. Target reveal is validation
    only, and is not a test condition unless it is added here before any test score exists.
  - **The proposal:** it specifies STACK's energy-distance alignment objective on (prompt,
    query, target) windows. Version 2 keeps that objective on the query positions and
    changes only what the revealed slots show, which the proposal does not fix.

- **Stage 1 (2026-11-23): final block + decoder, learning rate 3e-4.** Median validation
  wMSE of `shared_T1_diff` over the 21 validation doubles, 2,000 steps each:

  | scope | lr 1e-4 | lr 3e-4 |
  |---|---|---|
  | final block + decoder | 0.01959 | 0.01629 |
  | cross-cell attention in all layers + decoder | 0.01696 | 0.01615 |

  - **The tie:** cross-cell attention at 3e-4 is best, but the final block at 3e-4 is only
    0.91% above it, so the tie rule picks the smaller scope.
  - **Run-to-run noise:** training on the GPU is not bit-reproducible. Rerunning the final
    block at 1e-4 moved its median over the same 20 pairs by 1.1%. Per pair, the change was
    0.7% at the median and 3.7% at most. Gaps under about 1% are therefore within this
    noise.
  - **For reference:** frozen STACK (`icl_T1_diff`) scored 0.0260, the both-orders query
    cells 0.0222, and the additive baseline 0.0094.

- **Version 2, stage 1 (2026-11-24): cross-cell attention in all layers + decoder, learning
  rate 3e-4.** Median validation wMSE of `shared_T1_diff`, 2,000 steps each:

  | scope | lr 1e-4 | lr 3e-4 |
  |---|---|---|
  | final block + decoder | 0.02036 | 0.01773 |
  | cross-cell attention in all layers + decoder | 0.01789 | 0.01668 |

  - **No tie:** the next best, the final block at 3e-4, is 6.3% above the best.
  - **Against version 1:** every setting is worse, by 3% (all layers, 3e-4) to 9% (final
    block, 3e-4).

- **Version 2, stage 2 (2026-11-24).** Median validation wMSE, from the selected shared
  adapter for `shared_ttt` and from frozen STACK for `ttt`:

  | arm | 3e-5, 50 steps | 1e-4, 50 steps | 3e-4, 50 steps | 250 steps at the better rate |
  |---|---|---|---|---|
  | `ttt` | 0.02312 | 0.02256 | 0.02297 | 0.02231 (1e-4) |
  | `shared_ttt` | 0.01633 | 0.01615 | 0.01594 | 0.01544 (3e-4) |

  - **Both arms keep 250 steps:** 50 steps is 1.2% above for `ttt` and 3.2% above for
    `shared_ttt`, both outside the 1% tie. `ttt`'s margin is about the run-to-run noise.
  - **The random-pair arms** copy these settings.

- **No second test condition.** No condition was added before the test score existed (the
  test run ended at 06:07 on 24 November). Target reveal stays validation only.

- **The test run (2026-11-24)** used the settings above, once, on the 26 test doubles. The
  shared adapter was retrained on all 105 training doubles. Results are in the README
  ("Result on the 26 test doubles") and `results/ttt/test_v2/analysis.md`.
