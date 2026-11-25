# Experiment 2: Combinatorial perturbation prediction — plan

Owner: Reet · Branch: `reet` · Folder: `compositional/` · Dataset: Norman et al. (2019)
Companion docs: [`SPLITS.md`](SPLITS.md) (how data is divided for each question),
[`EVALUATION.md`](EVALUATION.md) (results), [`explore.ipynb`](explore.ipynb) (data exploration).

---

## In short

**The question.** In the Norman dataset, scientists switched on one gene (a "single
perturbation") or two genes at once (a "double perturbation") in human leukemia cells, then
measured how every gene's activity changed. STACK is an AI model that learns from examples it is
shown at prediction time (a "prompt"). We ask: if we show STACK cells with gene A switched on and
hand it cells with gene B switched on, can it predict cells with **both** A and B, even though it
was never trained to do that?

**Why it is hard.** STACK has to keep B's effect, add A's effect, and ideally capture how A and B
interact. Their combined effect is sometimes more, or less, than the sum of the parts.

**What we expect (Hypothesis 2).** Frozen STACK (as released, no extra training) will get some of
the "sum of the parts" right, but may:
- drop one of the two genes;
- miss interactions;
- give a different answer depending on which gene is shown as the prompt and which as the query.

Extra training (fine-tuning), and training that makes both orders agree, might help.

**How we judge it.** We compare against simple methods. The most important is just adding the
two single effects together, which is very strong on this dataset because most pairs really do
add up. So we look separately at the pairs that *don't* add up.

---

## Goals, hypothesis and questions

**Goal.** Test whether STACK can compose two separately observed perturbations in context
(Eq. composition): $\widehat{Q}_{AB} = f_\theta(P_A, P_{\mathrm{ctrl}}, Q_B)$.
- Prompt $P_A$: cells carrying A.
- $P_{\mathrm{ctrl}}$: matched unperturbed prompt cells.
- Query $Q_B$: cells carrying B.
- Target: measured A+B cells.

All effects are mean log-normalized expression minus the control mean.

**Hypothesis 2.**
- Frozen STACK recovers some additive effects, but fails to preserve both perturbations or to
  predict interactions.
- Predictions depend on prompt–query order.
- Fine-tuning, and an order-consistency objective, improve prediction of held-out combinations.

| # | question | models tested | needs training? | split ([SPLITS.md](SPLITS.md)) | key metrics |
|---|---|---|---|---|---|
| Q1 | Can frozen STACK compose at all? | stack_frozen (both orders), additive, weighted, prompt-only (δ_A), query-only (δ_B), control, mean, duplicates | weights only | **S-all** (131 doubles, weights cross-fitted) | wMSE, Pearson Δ, DE AUPRC, direction agreement |
| Q2 | Which behaviour: replacement, preservation, additive, or interaction? | stack_frozen per order | no | **S-all**, 262 pair-orders | β_prompt, β_query, residual Pearson |
| Q3 | Does order matter? Does averaging both orders help? | stack_frozen A→B, B→A, sym | no | **S-all**, paired per pair | paired Δ wMSE (A→B vs B→A), sym vs single order |
| Q4 | Does fine-tuning help on held-out combinations, and does it depend on genes seen in training? | stack_finetuned, ridge, kNN, PCA-linear, TabPFN, weighted, additive | yes | **S-gene** (5 seeds; seen 2/2, unseen 1/2, unseen 2/2) | wMSE, Pearson Δ, per category |
| Q5 | Does an order-consistency objective help? | stack_finetuned with vs without the consistency term | yes | **S-gene** (+ optional **S-order**) | wMSE, order gap |
| Q6 | Additive vs non-additive pairs: is r_AB recovered? | all | — | inside every split | residual Pearson, wMSE on non-additive pairs |
| ref | Comparability with the team and Study 4 | all | — | **S-shared** (seed-0, 26 doubles) | all shared metrics |

**Baselines** (from the shared Baselines section):
- negative controls: control and mean;
- positive controls: technical and interpolated duplicate (team halves convention, see below);
- direct transfer: prompt-only δ_A (the "replacement" reference) and query-only δ_B (the
  "preservation" reference);
- additive;
- ridge, kNN and PCA-linear;
- TabPFN.

Supervised baselines get the same training doubles and singles as fine-tuned STACK.

**Metrics:** Pearson Δctrl, DE AUPRC, direction agreement, wMSE (primary), Spearman of log-fold
changes, top-gene recovery. All are calibrated against the positive and negative controls (DRF).
Each is reported on the 2,000 HVG panel (chosen without held-out responses) and on the full STACK
gene panel. 95% CIs come from a paired bootstrap over held-out combinations.

**Truth / duplicate halves (decided).**
- Each condition's cells are split 50/50 at random (`splits/norman2019_halves.csv.gz`, the team
  convention).
- Models are scored against the first half.
- The second half is the **technical duplicate**: "predicting" one half of the real cells from the
  other half shows the best score achievable given measurement noise. That makes it a ceiling to
  compare models against.

**Two scoring modes (every model is graded both ways).**

*In short:* the "right answer" for a held-out pair is the average of its measured cells. We grade
in two ways:
- **Halves (main result):** grade against a random half of the cells. The other half acts as a
  repeat experiment, showing the best score achievable given noise.
- **All cells (robustness check):** grade against all the cells. This is less noisy, especially for
  small pairs, but leaves no repeat experiment to compare against.

If both rank the models the same way, our conclusions don't depend on the halving.

*Technical:*

| piece | halves | all cells |
|---|---|---|
| truth profile | mean of first-half cells | mean of all cells |
| DE vs rest (t-test), DE vs control (Wilcoxon) | first half | all cells |
| control, mean, additive baselines; model inputs | identical | identical |
| technical / interpolated duplicate | yes | not definable; shown as "n/a" |
| gap closed / DRF | vs negative controls; DRF from the duplicate | gap closed only |
| non-additive flags | the same flags in both modes, so subsets match | ← |

- Code: `stack_compositional.contexts.build_context(..., truth="halves"|"all")` and
  `scripts/build_contexts.py`.
- Contexts are saved to `data/processed/compositional/<split>_{halves,all}.npz`.
- Scores carry a `truth` column.
- `make_table.py` writes both modes to `EVALUATION.md`, plus the LaTeX tables
  `composition_results.tex` (halves) and `composition_results_allcells.tex`.

---

## Technical details

### Shared setup we reuse (do not change)
| what | where |
|---|---|
| data loading, log-normalization | `scripts/norman_baselines.py`, `stack_robustness.data` |
| truth / duplicate halves | `stack_robustness.baselines.assign_halves`, `splits/norman2019_halves.csv.gz` |
| metrics (12), gap-closed / DRF | `stack_robustness.scoring`, `.metrics` |
| scoring any prediction table | `stack_robustness.evaluation.EvalContext.score` |
| additive, mean, control baselines, interpolated duplicate | `stack_robustness.baselines` |

`ttt/` (Study 4) stays on the branch but is not imported or run. STACK generation code lives in
`compositional/`; `ttt/src/stack_ttt/predict.py` may be used as a reference.

### Arms
| arm | definition | GPU |
|---|---|---|
| `additive` | $\delta_A + \delta_B$ (measured singles) | no |
| `prompt_only`, `query_only` | $\delta_A$; $\delta_B$ | no |
| `weighted` | $w_A\hat\delta_A + w_B\hat\delta_B$, $\hat\delta_X = f(P_X, P_{ctrl}, Q_{ctrl})$; $w\in\{0.1,0.3,…,0.9,1.0,1.2\}^2$ chosen on training doubles only | yes |
| `ridge` | per gene: $\delta_{AB} \approx \beta_1(\delta_A+\delta_B) + \beta_2\,\delta_A\delta_B + \beta_0$, λ by CV on training doubles | no |
| `knn`, `pca_linear` | kNN on single-effect features; linear model in PCA space (Ahlmann-Eltze et al.) | no |
| `stack_frozen` | $f(P_A, P_{ctrl}, Q_B)$ per order | yes |
| `stack_sym` | $\tfrac12[f(P_A,P_{ctrl},Q_B) + f(P_B,P_{ctrl},Q_A)]$ (Eq. symmetric) | yes |
| `stack_finetuned` (± consistency) | one adapter trained on training doubles, both orders (+ λ·‖f(A→B) − f(B→A)‖²) | yes |

STACK outputs are scored "diff" style, as the STACK tutorial does: the observed query profile,
plus generation with the prompt, minus generation with a control prompt, with seeds paired. Raw
scores are kept as well.

### Non-additive analysis
- $r_{AB} = \delta_{AB} - (\delta_A + \delta_B)$; $\hat r_{AB} = \hat\delta_{AB} - (\delta_A+\delta_B)$.
- $\mathrm{MSE}(\hat r, r) = \mathrm{MSE}(\hat\delta, \delta)$ exactly, so we report **residual
  Pearson**, $\mathrm{corr}(\hat r, r)$, on each pair's top-100 DE genes.
- Additive / buffering-synergy / interaction groups come from the additivity ruler (calibrated wMSE of the plain and stretched sums); see SPLITS.md.

### Behaviour classification (Q2)
- Per pair-order, fit by least squares on the top-200 DE genes:
  $\hat\delta_{AB} \approx \beta_P\delta_{prompt} + \beta_Q\delta_{query}$.
- (1, 0) → replacement; (0, 1) → preservation; (1, 1) → additive; residual Pearson → interaction.
- The same fit on the truth gives the reference: Norman et al.'s $c_A, c_B$ plot.

### Evaluation table (auto-updated)
`scripts/make_table.py` reads scores and rewrites `EVALUATION.md` and `tables/composition_results.tex`.

### Things to raise with the team
1. Study 4's `icl` arm uses the same setup as `stack_frozen`. Agree who reports it.
2. The `main.tex` weight grid is {0.1…0.9}, but the text calls additive the w = 1 case. Our grid
   includes 1.0.
3. Study 4 combines the two orders with a median; Exp. 2 uses a mean (as written).
4. "Residual error" measured as MSE would duplicate the MSE column, so we report residual Pearson.
5. Study 4's list of 9 "non-additive" pairs ranks pairs by additive wMSE on the test truth. It
   includes JUN+CEBPA (54 cells, residual share 0.035) and leaves out CEBPB+PTPN12 (0.36). We
   propose a noise-aware definition instead (SPLITS.md).
6. Environment: `envs/.venv` is Python 3.14, and the shared tests pass on it.

---

## Steps (one at a time, approval after each)

| # | step | GPU / data | status |
|---|---|---|---|
| 0 | Branch `reet` up to date with shared code; `envs/.venv` set up; 57 shared tests pass | no | done |
| 1 | This plan | no | done |
| 2 | Folder skeleton, `EVALUATION.md` + LaTeX table, `report.py` + `make_table.py` (+5 tests) | no | done |
| 2.5 | `../data/` folder; download Norman + vocab; first `explore.ipynb` | data | done |
| 2.6 | Split design (`SPLITS.md`), goals and questions (this file), `explore.ipynb` by question | data, CPU | done, awaiting review |
| 2.6b | Second scoring mode, truth from all cells: `contexts.py`, `build_contexts.py`, both modes in `make_table.py`, tests; checked on S-shared | data, CPU | done |
| 2.6d | Additivity ruler (`additivity.py` + tests; notebook Part B / Q2 / Q6 on scoring genes) | data, CPU | done |
| 2.6e | Cutoff agreed: top 25% of the plain-sum score = non-additive; same cut splits buffering/synergy vs interaction → 98 / 23 / 10 (`splits/compositional/additivity_groups.csv`); groups wired into `make_table.py` | data, CPU | done |
| 2.7 | Generate split files: `splits/compositional/` (S-all folds, S-gene seeds, non-additive flags) + tests | data, CPU | done |
| 3 | `combine.py` (additive, weighted, symmetric, residual, ridge, kNN, PCA-linear) + tests | no | |
| 4 | `behavior.py` + tests | no | |
| 5 | Scoring contexts per split; CPU baselines → first table rows | data, CPU | |
| 6 | `stack_run.py`: STACK generation + diff scoring; smoke test on 3 pairs | GPU | |
| 7 | Q1–Q3 on S-all (frozen STACK, both orders, singles for `weighted`) | GPU | |
| 8 | Q4–Q5 on S-gene: fine-tuned STACK (± consistency); validation, then freeze `PROTOCOL.md`, then test | GPU | |
| 9 | Q6 analysis, S-shared reference, `make_table.py`, results into `README.md`; commit | no | |

### How we check it works
- Unit tests: weighted with w = (1, 1) equals additive; the residual of the additive prediction is
  0; symmetric of identical inputs returns the input; ridge recovers additive coefficients on
  synthetic additive data; the behaviour classifier labels synthetic cases correctly; no split leaks
  test doubles into training or validation.
- Our `additive` row on S-shared must equal the shared baseline in `results/norman_baselines/scores.csv`.
- Frozen STACK on S-shared should be close to Study 4's `icl` row (median wMSE ≈ 0.017).
