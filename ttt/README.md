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
- **Not yet measured:** whether frozen STACK carries a Norman prompt onto control query
  cells, the layout the paper uses.

**A mismatch between adapter training and inference.**
- **At inference,** arc-stack fills the slots between the prompt and the query with more
  prompt cells (`get_prediction`). At T = 1, the first 65% of each window is prompt.
- **In STACK's post-training,** those slots hold revealed true cells, and they share the
  prompt's condition. So its training and inference agree.
- **This study's pair windows** reveal true A+B cells in those slots
  (`objective.ttt_loss`), and those do not share the prompt's condition, A. An adapter can
  read the target from them in training, then meets A cells there at inference.
- **The validation loss curves** were measured with A+B cells revealed, so they do not
  show that an adapter learned to use the prompt.
- **Before the test run,** inference with no context slots (`context_ratio` 0) will show
  whether this matters.

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
  - the same Norman count matrix (13,815 cells, no column differs);
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

Built and unit-tested: everything above.

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

Next steps:
1. Reconstruct Arc's tutorial PBMC data through this code, to rule the pipeline in or out.
2. A validation `EvalContext` on doubles carved from the training doubles. Hyperparameters
   are chosen there; no cell of a held-out double enters adaptation or selection.
3. An adapter trained on random non-neighbour doubles, the control TTT must beat, then
   all 26 held-out doubles. That fits on this machine's GPU in a few hours.
4. Energy distance alone at T=1 against the full objective, f_CLS and T=5.

Deferred:
- The X-Atlas/Orion arm, which needs a slice of the 126 GB Hugging Face copy.
- k-nearest-neighbour retrieval of H(g*).
- The shared context adapter. The null-only adapter is the first such control.
- Count-space input augmentations.
