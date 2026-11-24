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

## Result on the 26 test doubles (version 2)

**TTT on a pair's neighbours adds a small, consistent gain on top of the shared adapter, but
no STACK variant comes near the additive baseline.**
- **The run:** one run on 24 November 2026, with the settings validation chose (PROTOCOL.md,
  Decisions).
- **The settings:** the shared adapter on cross-cell attention in all layers + decoder at
  3e-4, and TTT for 250 steps, at 1e-4 from frozen STACK and at 3e-4 from the shared adapter.
- **Full tables:** `results/ttt/test_v2/analysis.md`.

Median wMSE on the STACK gene panel, diff-scored, T = 1 (lower is better):

| prediction | all 26 | without pilot (23) | non-additive (9) |
|---|---|---|---|
| frozen STACK (`icl`) | 0.0168 | 0.0163 | 0.0376 |
| TTT from frozen STACK (`ttt`) | 0.0165 | 0.0143 | 0.0371 |
| the same TTT on random pairs (`ttt_random`) | 0.0177 | 0.0160 | 0.0371 |
| shared adapter (`shared`) | 0.0128 | 0.0116 | 0.0335 |
| shared adapter + TTT (`shared_ttt`) | 0.0108 | 0.0106 | 0.0314 |
| shared adapter + TTT on random pairs (`shared_ttt_random`) | 0.0138 | 0.0109 | 0.0355 |
| query cells, unchanged (both orders) | 0.0155 | 0.0141 | 0.0347 |
| additive baseline | 0.0068 | 0.0064 | 0.0114 |
| interpolated duplicate | 0.0052 | 0.0056 | 0.0041 |

Paired differences over all 26 pairs. Each is B minus A per pair, so positive means A is
better, with a bootstrap 95% interval for the median.

| A vs B | median | 95% interval | A better on |
|---|---|---|---|
| `shared_ttt` vs `shared` | +0.0009 | [+0.0004, +0.0012] | 20 of 26 |
| `shared` vs `icl` | +0.0036 | [+0.0011, +0.0072] | 21 of 26 |
| `shared_ttt` vs `shared_ttt_random` | +0.0008 | [+0.0001, +0.0013] | 18 of 26 (1 tie) |
| `ttt` vs `icl` | +0.0009 | [+0.0001, +0.0014] | 19 of 26 |
| `ttt` vs `ttt_random` | +0.0001 | [−0.0002, +0.0005] | 15 of 26 (1 tie) |
| `icl` vs query cells | −0.0015 | [−0.0032, −0.0007] | 3 of 26 |
| `shared_ttt` vs query cells | +0.0030 | [+0.0002, +0.0054] | 18 of 26 |
| `shared_ttt` vs additive | −0.0057 | [−0.0131, −0.0017] | 4 of 26 |

The tie is KIF18B+KIF2C. It has no neighbours, so both TTT arms run the same tasks on it.

**Clear** (the interval excludes zero, with or without the pilot pairs):
- **TTT improves the shared adapter.** TTT on neighbours lowers its error by a median
  0.0009 per pair, about 6%, on 20 of 26 pairs.
- **The shared adapter gives most of the gain.** It improves on frozen STACK by four times
  as much (0.0036, on 21 of 26 pairs).
- **Frozen STACK is worse than doing nothing.** Leaving the query cells unchanged beats it on
  23 of 26 pairs.
- **The additive baseline beats every STACK arm.** The best arm, `shared_ttt`, beats it on 4
  of 26 pairs. Its median gap (0.0057) is six times TTT's gain.

**Suggestive only** (the interval's lower end is 0.0001 to 0.0002):
- **Neighbours over random pairs, from the shared adapter:** neighbours win by 0.0008, on 18
  of the 25 pairs that differ. The gain may come partly from related pairs, but the evidence
  is weak.
- **TTT from frozen STACK:** it improves on frozen STACK (0.0009, 19 of 26). TTT on random
  pairs does as well (next list), so this gain does not come from the neighbours.
- **`shared_ttt` against the unchanged query cells:** it wins by 0.0030, on 18 of 26 pairs.

**No effect:**
- **Neighbours against random pairs, from frozen STACK:** +0.0001 [−0.0002, +0.0005].
- **The spec's test:** TTT's gain does not track the additive baseline's error. The
  Spearman correlation is ρ = 0.18 (p = 0.37) for `shared_ttt` over `shared`, and
  ρ = 0.22 (p = 0.28) for `ttt` over `icl`.
- **T = 5 against T = 1:** no median moves by more than 0.0008.

**Too few pairs to tell:** on the 9 non-additive pairs, every TTT interval includes zero. That
does not show TTT fails there.

**Prompt use** (exploratory, 52 pair-orders): the shift that A's prompt causes correlates
with A's own effect at a median of:
- −0.24 for frozen STACK;
- +0.03 for `ttt`, with 15 of 52 still below −0.3, so TTT alone does not teach STACK to
  follow the prompt;
- +0.57 for `shared` and +0.60 for `shared_ttt`. The shared adapter is what makes it follow.

**Cost per pair** (medians from `pairs/*.json`):
- **Per arm:** TTT's 250 steps take 93 s. Generating takes 7 s at T = 1 and 27 s at T = 5.
- **So:** TTT makes a T = 1 prediction about 14 times slower.
- **The shared adapter** trains once, in 837 s on 211 tasks.
- **GPU memory:** the whole job peaked at 5.4 GB, as dibs measured it. Per-arm peaks were not
  saved.

**Limitations:**
- **One run each:** one shared adapter, and one seed per arm. The intervals resample pairs,
  not training runs. On validation, retraining moved a median by 1.1%.
- **Selections at the edge of the grid:** the shared adapter took the larger scope and the
  higher rate, and `shared_ttt` the higher rate and more steps. So neither is shown to be at
  its best. `ttt`'s 250 steps beat 50 by 1.2% on validation, about the run-to-run noise.
- **No new information:** the neighbours are already in the shared adapter's training set.
  TTT on them re-specialises the model.
- **No multiple-comparison correction:** the protocol registered none.
- **Sign tests, added after the test** (two-sided, ties dropped):
  - `shared_ttt` vs `shared`: p = 0.009;
  - `shared_ttt` vs `shared_ttt_random`: p = 0.043;
  - `ttt` vs `icl`: p = 0.029;
  - `shared` vs `icl`: p = 0.002;
  - `ttt` vs `ttt_random`: p = 0.42.
- **The pilot pairs:** the 3 pilot pairs informed the method (Status). Without them, the
  conclusions are the same.

**What is committed:** each run's `analysis.md`, `summary.md`, `scores.csv`, `args.json`,
loss history and per-pair JSON. The predicted profiles (`pairs/*.npz`) and adapter weights
(`*.pt`) stay local.

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
  perturbations' cells are split into disjoint "adapt" and "infer" halves. Neighbour
  tasks draw on the adapt halves and prediction starts from the infer halves, so no task
  pairs the cells a prediction starts from with a target. The held-out double's own
  infer cells enter adaptation only through masked reconstruction, as the spec allows.
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

## STACK's paper and code, against this study

**What was read.**
- **The paper:** Dong et al., *Stack: In-Context Learning of Single-Cell Biology*,
  bioRxiv (2026), doi:10.64898/2026.01.09.698608. This covers the main text and Methods,
  from a text extraction that lacks the tables, Box 1, equations, figures and supplement.
  "The paper does not" below means "the main text and Methods do not".
- **The code:** github.com/ArcInstitute/stack at v0.1.3, which `uv sync` installs; GitHub
  main differs only in its README.
  - Read in full: `models/finetune/model.py`, `models/finetune/mixins.py`,
    `finetune/lightning.py`, `configs/finetuning/ft_parsecg.yaml`, and three functions of
    `models/core/inference.py`: `get_prediction`, `get_incontext_prediction` and
    `get_incontext_generation`.
  - Read in part: `data/finetuning/datasets.py` (window construction), `models/core/base.py`,
    `models/utils.py`, `cli/generation.py`, the prediction tutorial and the README.

**Where this study follows them.**
- **Diff scoring is the paper's "synthetic control".** For perturbational in-context
  tasks, the paper also generates from the unperturbed prompt sample and uses that as the
  reference (Results, section 2.3; Methods, "Evaluation data construction").
- **Generation uses Arc's CLI settings.** These are `mdm` mode, 25% prompt and context
  rising from 0.2 to 0.4 (`cli/generation.py`). The Python function's own default is
  `vanilla`, which re-predicts every cell at each step.
- **The adaptation loss is STACK's post-training loss** (`models/finetune/model.py`,
  `mixins.py`):
  - 10–30% of genes masked;
  - a revealed share of the non-prompt positions, up to 0.9375;
  - energy distance on the 1,000 genes with the largest Pearson-residual variance and on
    teacher embeddings, divided by 1 − t;
  - reconstruction on prompt and revealed cells;
  - sliced Wasserstein (weight 0.01) and the classifier's cross-entropy.
- **The teacher is called as `finetune/lightning.py` calls it.** It sees the true
  window, unmasked, with every position treated as a prompt cell.
- **The checkpoint's settings match the paper's post-training.** Dropout is 0, there are
  128 prompt cells per 512-cell window, learning rate 2e-5 and weight decay 0.003.
- **Repeating a small query set has precedent.** The paper upsampled query cells to at
  least 2,000 or 1,000 per cell type in two of its tasks.

**Where it departs.**
- **Generation steps.** The headline here is T = 1 for cost, with T = 5 beside it.
  The paper's in-context benchmarks use T = 5 (Fig. 3 caption) and report modest gains
  from several steps over one (Fig. S9).
- **Combining perturbations.** The paper's method for pairs is: "For combinatorial
  perturbation evaluations, we computed weighted sums of single-perturbation Stack
  predictions", with weights chosen by grid search. It does not prompt with one
  perturbation and query cells carrying another. That is this study's own use of STACK.
- **Prompt size.** For its Perturb Sapiens atlas, the paper keeps only conditions with
  more than 332 cells, "the minimum needed to fill one prompt cell set at the final Stack
  generation step". It states no floor for its benchmarks.
  - Here a prompt is one single's infer half, median 248 cells over the 105 singles.
  - 21 of the 42 validation prompts, and 29 of the 52 test prompts, are smaller than 333.
    arc-stack then repeats their cells within a window (the modulo over `base_indices` in
    `get_incontext_prediction`).
  - The control prompt (2,963 cells) never repeats.
- **The cells STACK was aligned on.** Post-training used 45M cells from CELLxGENE
  datasets of over 50,000 cells and at least five donors, and Parse's 10M PBMCs. Its
  limitations section says the model "is post-trained primarily on in vivo cell types,
  especially immune cells". Norman's K562 cells are an in vitro leukemia line.
- **Gene names.** Post-training read CELLxGENE genes by `feature_name`, its current
  symbols (`configs/finetuning/ft_parsecg.yaml`). The vocabulary is the union of each
  scBaseCount file's top 1,000 variable genes, and 392 of its names are symbols HGNC has
  since renamed. That is consistent with those genes (H3F3A, H2AFZ, ...) collapsing in
  every dataset tried.
- **Training.**
  - Post-training updated every weight with AdamW at 2e-5, an effective batch of 32 and
    bf16. It moved the teacher 5% toward the student every 500 steps.
  - Here only the adapters and f_CLS train, one window per step, in fp32, with the teacher
    frozen. Arc's README calls its procedure "frozen teacher" fine-tuning.
- **Confidence filtering.** Perturb Sapiens drops generated cells whose classifier logit
  is above 2.5, and the paper does not say whether its benchmarks did. Here every
  generated cell is kept.

**What frozen STACK does with a Norman prompt** (validation, T = 1, measured on saved
predictions). These numbers come from the 21 validation doubles in both prompt orders, 42
pair-orders. Query cells there carry the other single perturbation.
- **Measure:** compare generating with A's prompt against generating with a control prompt.
- **The shift is small and runs the wrong way.** Its size is a median 14% of A's own effect
  on A's top 200 genes. Its median correlation with A's effect is −0.23: 19 of 42 are below
  −0.3, and 2 are above +0.3.
- **Prompt size doesn't change it.** Prompts of 333 or more cells give a pooled
  coefficient of −0.03, and smaller ones −0.01.
- **The three pilot pairs behave the same at T = 5.**
- **The generated profile keeps part of the query's own perturbation.** Pooled over
  genes, it is 0.30 × the query's effect − 0.01 × the prompt's (R² 0.21).
- **The shared adapter (final block, 1e-4) turns the shift toward A's effect,** with a
  median correlation of +0.34 (23 of 42 above +0.3). The shift stays small, at 21% of the
  effect size.
- **The paper's own layout gives the same answer.**
  - Setup: each validation single's prompt onto 537 control query cells, against a control
    prompt onto the same cells, at T = 1 and context 0.4 (`scripts/prompt_checks.py`).
  - Result: the median correlation with the prompt's effect is −0.25, with 0 of 31 prompts
    above +0.3 and 15 below −0.3, and a median size of 14%.
  - So frozen STACK does not read Norman's perturbations from a prompt at all. It is not
    only failing to compose them onto perturbed cells.

**Two layouts for the slots after the prompt.**
- **At inference,** arc-stack fills the slots between the prompt and the query with more
  prompt cells (`get_prediction`). At T = 1, the first 65% of each window is prompt.
- **In STACK's post-training,** those slots hold revealed true cells, and they share the
  prompt's condition. So its training and inference agree.
- **Version 1's pair windows** revealed true A+B cells in those slots (`objective.ttt_loss`),
  which do not share the prompt's condition, A. An adapter could read the target from them
  in training, then met A cells there at inference.
- **The validation loss curves** were measured with A+B cells revealed, so they do not
  show that an adapter learned to use the prompt.
- **That did not hold version 1 back, by two measurements on validation:**
  - **Without the slots:** dropping them at inference (`context_ratio` 0,
    `scripts/prompt_checks.py`) made version 1's shared adapter worse. The error rose from
    0.0163 to 0.0185, and the shift fell from 31% to 18% of A's effect. Frozen STACK moved
    from 0.0260 to 0.0222, its shift away from the prompt shrinking from 14% to 7%.
  - **Version 2's layout:** it is the inference-consistent one, revealing prompt cells as
    inference does (PROTOCOL.md, Changes). Its first shared adapter (final block, 3e-4)
    scored 0.0177 against version 1's 0.0163, with a prompt correlation of +0.58 against
    +0.67 and a shift of 24% against 31%. Over all four stage-1 settings, version 2 is 3%
    to 9% worse than version 1 (PROTOCOL.md, Decisions).
  - **Two confounds:** version 2 also renames genes and freezes f_CLS. A validation-only run
    with the old layout and version 2's other settings (`results/ttt/val_v2_target`, final
    block, 3e-4) isolates the layout. The renaming and f_CLS stay confounded with each other.
- **All STACK predictions are smaller than the additive one.** On the additive prediction's
  top 200 genes, their effects correlate with it at about 0.97. Their size relative to it is
  49% for frozen STACK (about what B's observed profile alone gives), 66% for version 1's
  shared adapter and 62% for version 2's.

## STACK does not reproduce Norman's cells (and what the comparison does about it)

Run frozen on Norman's K562 cells, STACK shifts hundreds of genes away from the data
whatever the prompt. This happens even when it only reconstructs fully observed control
cells, with nothing masked or generated. In those cells, MALAT1 drops from 4.6 to 1.5 in
log-normalized expression and H3F3A, HIST1H4C and H2AFZ go from about 2.9 to 0.01, while
HBG2 rises from 1.8 to 4.5. The top 1,000 genes carry 90% of the squared error.

What is known about the cause:
- Genes whose vocabulary symbol HGNC has since renamed are 2.7% of the panel but carry
  15.6% of the error, including every histone gene that collapses. Most of the error
  (84%) is in genes with current symbols, such as MALAT1, the hemoglobins, B2M and ACTG1.
- Mapping 713 renamed Norman genes into vocabulary slots they were missing from
  (`stack_robustness.genes.vocabulary_aliases`) does not change it.
- This pipeline is not the cause. `scripts/pipeline_check.py` (output in
  `results/ttt/pipeline_check/`) finds it identical to Arc's own code:
  - the same weights (largest difference 0.0);
  - the same Norman count matrix (13,815 cells, no column differs), before version 2's
    renaming of 713 genes into the vocabulary (PROTOCOL.md, Changes) adds their columns;
  - the same generated cells for the same seed, on Norman and on Arc's tutorial data
    (`Stack-DrugPBMC-Example`).
- STACK shifts Arc's tutorial data too, but less. In both checks below (T = 1), STACK generates
  the query cells, and the generated profile is compared with the query cells' observed
  profile. The query cells are untreated, so the two should agree.

  | data | prompt | query | correlation | genes shifted by more than 0.5 |
  |---|---|---|---|---|
  | Norman | control K562 cells | other control K562 cells | 0.84 | 234 |
  | tutorial PBMCs | DMSO T cells | DMSO myeloid and B cells | 0.95 | 77 |

  STACK was aligned partly on PBMCs; K562 leukemia cells are further from what it has
  seen. The same histone symbols collapse in the tutorial data (H3F3A, H3F3B and H2AFZ go
  to about 0), so that part is STACK's whatever the data.
- On the tutorial's own task, STACK about ties a baseline with no model. The task is to
  predict each drug's effect on myeloid and B cells from drug-treated T cells (T = 5, as
  Arc's CLI). The baseline copies the effect the T cells show.

  | over 14 drugs (median) | STACK | copy the T cells' effect |
  |---|---|---|
  | Pearson correlation | 0.41 | 0.39 |
  | Pearson correlation, each drug's top 200 genes | 0.52 | 0.58 |

  STACK does better on 6 of 14 drugs by correlation, and on 4 of 14 on the top 200 genes.
  STACK is scored here as the tutorial's synthetic control does: generated with the drug
  prompt minus generated with a DMSO prompt. Without that subtraction its correlation
  falls to 0.27, and it is worse on 11 of 14 drugs.

  The paper reports an advantage for STACK on this dataset (Fig. 3E). Its evaluation is
  not the same as this check's:

  | | the paper | this check |
  |---|---|---|
  | metrics | Cell-Eval, on the top 2,000 variable genes | Pearson correlation over every expressed gene |
  | query cells | upsampled per cell type | as measured |
  | prompt | its own task construction | T cells only, mostly 318–533 cells (two of 14 under 333) |

Raw scoring charges this shift to STACK, and an adapter can "improve" just by learning it.
`norman_compare.py` therefore reports, side by side:
- `*_raw`: the generated profile, scored as Eq. 1 specifies.
- `*_diff`: the observed query profile, plus generation with the prompt minus generation
  with a control prompt (seeds paired, separate control cells). It cancels shifts STACK
  applies whatever the prompt. It is consistent with Cell-Eval's predicted effect
  (predicted minus predicted control) if the predicted controls are generated too.
  Under diff scoring, report the margin over `query`.
- `query`: the query cells unchanged, with no model. In one order (`query_one`) this is
  B's own measured profile, which STACK should not get credit for. The both-orders row
  averages A's and B's profiles, so its change from control is roughly half the additive
  prediction. Its Pearson Δctrl nearly matches the additive baseline's, since that
  correlation ignores scale; Δpert does not (0.666 against 0.792 in the pilot).
- `nulladapt_*`: one adapter trained only on control → control. It also learns "zero
  effect", so it is a biased control in raw space; read it on the diff rows.

Which of raw and diff becomes the headline metric is a decision for the team.

## Norman neighbourhoods

For the 26 held-out doubles (seed 0), H(g*) has 0 to 15 training doubles; most have 3
to 15. Three are thin: KIF18B+KIF2C has 0, TMSB4X+BAK1 has 1 and SGK1+S1PR2 has 2.
These lean on the null task (control → control) and on reconstruction of their own
prompt cells. JUN+CEBPA has only 54 cells.

## Status

Built and unit-tested: everything above. Run: the pilot, the pipeline check against Arc's
tutorial, validation in two versions, and the test run ("Result on the 26 test doubles").

Pilot (`results/ttt/norman_pilot3`): 3 held-out doubles, default STACK loss weights, no
tuning. It is a pipeline and cost check, not a result. Per double, frozen prediction (both
orders, 2 draws, with the control-prompt runs) takes 4 to 8 s at T=1 and 15 to 32 s at
T=5, and 50 adaptation steps about 13 s, with 1.5 GB of GPU memory for adaptation and
3.1 GB at peak. On these 3 doubles, no STACK variant beats the unchanged query cells
or the additive baseline, frozen or adapted, raw or diff. The null-only adapter matches
most of TTT's raw-MSE gain, so that gain is largely fitting STACK to the dataset.

Disclosure: the pilot's 3 doubles (AHR+FEV, AHR+KLF1, BPGM+SAMD1) are held-out test
doubles, and they have informed the method choices above. All further tuning and
diagnosis uses validation doubles.

Open:
- **The layout run on validation** (`results/ttt/val_v2_target`) is queued. It tests whether
  version 2's layout accounts for its gap to version 1 ("Two layouts for the slots after the
  prompt").
- **The remote HPC cluster's package versions** are newer than this lockfile's: pandas 3.1.1,
  anndata 0.13.4, scvi-tools 1.5.2, and torch 2.14.1 with CUDA 12.6. The code has not been
  run under them yet.

Deferred:
- The X-Atlas/Orion arm, which needs a slice of the 126 GB Hugging Face copy.
- k-nearest-neighbour retrieval of H(g*).
- Energy distance alone against the full objective.
- Count-space input augmentations.
