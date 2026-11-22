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
- Neither this pipeline nor gene naming is ruled out. The next check is to reconstruct
  Arc's own tutorial data (`Stack-DrugPBMC-Example`) through the same code.

Raw scoring charges this shift to STACK, and an adapter can "improve" just by learning it.
`norman_compare.py` therefore reports, side by side:
- `*_raw`: the generated profile, scored as Eq. 1 specifies.
- `*_diff`: the observed query profile, plus generation with the prompt minus generation
  with a control prompt (seeds paired, separate control cells). It cancels shifts STACK
  applies whatever the prompt. It is consistent with Cell-Eval's predicted effect
  (predicted minus predicted control) if the predicted controls are generated too.
  Under diff scoring, report the margin over `query`.
- `query`: the query cells unchanged, with no model. For a double this is B's own
  measured effect, which STACK should not get credit for.
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
tuning. It is a pipeline and cost check, not a result. Per double, frozen prediction takes
2 to 20 s and 50 adaptation steps about 15 s, with 1.5 GB of GPU memory for adaptation
and 3.1 GB at peak. On these 3 doubles, no STACK variant beats the unchanged query cells
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
