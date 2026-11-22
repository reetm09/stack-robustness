# stack-robustness
CS 282 Final Project

Shared scoring code for the four studies in `proposal/main.tex`: the target effect of
Eq. 1, the positive and negative controls, the metrics, and the dynamic range fraction
(DRF). Every model, whether STACK, TabPFN or a baseline, is scored by the same code, on
the same held-out cells and gene panels.

## Setup

```bash
uv sync                # Python 3.12 environment from pyproject.toml
uv run pytest          # unit tests
```

Data goes in `data/` (git-ignored). Norman et al. (2019) and STACK's gene list:

```bash
mkdir -p data/raw/norman data/raw/stack_vocab
curl -L -o data/raw/norman/NormanWeissman2019_filtered.h5ad \
  "https://zenodo.org/api/records/13350497/files/NormanWeissman2019_filtered.h5ad/content"   # scPerturb, 699 MB
curl -L -o data/raw/stack_vocab/basecount_1000per_15000max.pkl \
  "https://huggingface.co/arcinstitute/Stack-Large-Aligned/resolve/main/basecount_1000per_15000max.pkl"
```

## Scoring a model

A prediction is a table of pseudobulk profiles: perturbations × genes, the mean
log-normalized expression (`log1p(counts / total * 1e4)`) of the predicted perturbed
query cells. For a model that generates cells, such as STACK, average its generated
cells first.

STACK's generator (`get_incontext_generation` in `arc-stack`) returns counts
sampled from its negative-binomial output over its 15,012 genes, at each query cell's
library size. Log-normalize them with the query cells' all-gene totals before
averaging, the same convention as the ground truth (scPerturb's `ncounts`). Otherwise
the predictions carry a systematic scale offset. Score the samples, not the NB means:
`log1p` of means runs high against the truth. `ttt/src/stack_ttt/predict.py` does
both steps.

```python
from stack_robustness.scoring import score_profiles, relative_to_negative

scores = score_profiles(pred, truth, ctrl, pert_ref, de_vs_ctrl, de_vs_rest)  # perturbations x metrics
gaps = relative_to_negative({"stack": scores, **baseline_scores},
                            mean_baseline="mean_baseline", control_baseline="control_baseline")
```

`scripts/norman_baselines.py` shows how each input is built.

## Definitions

These follow the code of Miller et al.
([shiftbioscience/Perturbation-Models-Outperform-Baselines](https://github.com/shiftbioscience/Perturbation-Models-Outperform-Baselines))
unless stated otherwise.

- **Truth and duplicate halves.** Each condition's cells are split at random into
  halves (`assign_halves`, fixed seed). Every model is scored against the **first**
  half. The **second** half is the technical duplicate.
- **Negative controls.** `control_baseline` predicts the mean control profile.
  `mean_baseline` predicts the average of the training perturbations' profiles, each
  perturbation weighted equally.
- **Positive controls.** `technical_duplicate` is the second half's mean.
  `interpolated_duplicate` is `α·duplicate + (1−α)·mean_baseline` per gene, with
  `α = 1 − p_adj`. The p-values come from a t-test of the second half against all
  other cells, as in Miller et al.
- **Other baselines.** `additive` is the control plus each single perturbation's effect.
  `source_effect_copy` is the prompt context's effect moved onto the query controls.
- **DE genes.** There are two DE tests, both run on the truth half:
  - Against control cells (Wilcoxon; the proposal's choice for Study 2). This defines
    the true DE genes (adjusted p < 0.05) for the Cell-Eval-style metrics.
  - Against all other perturbed cells, with control cells left out (t-test with
    overestimated variance, as in Miller et al.). This gives the gene weights for
    `wmse` (squared, min-max scaled |t|) and the top 100 genes for the `_top` Pearson
    metrics. Both are computed within the panel being scored, so each panel's scores
    depend only on its own genes. Miller et al. use one fixed gene set.
- **Metrics** (`scoring.METRICS`):
  - `pearson_deltactrl`: Pearson correlation of the predicted and true change from
    control.
  - `spearman_deltactrl_de`, `direction_agreement`: the Spearman correlation and the
    sign agreement of those changes, on the true DE genes.
  - `de_pr_auc`: average precision of |predicted change| as a score for the true DE
    genes.
  - `top_overlap`: the share of the 20 top true DE genes found among the 20 largest
    predicted changes.
  - `mse`, `mae`, `wmse`: unweighted and weighted errors.
  - `pearson_deltapert`, `pearson_deltapert_top`: Pearson correlation of changes
    measured from the mean baseline instead of from control.
  - `pearson_deltactrl_top`: `pearson_deltactrl` on the top 100 DE genes.
  - `nir`: the share of other held-out perturbations whose truth is farther from this
    prediction than its own truth is.
- **DRF and gap closed.** `relative_to_negative` gives, per perturbation and metric,
  the share of the gap between the metric's negative control and a perfect score that a
  model closes, clipped to [−1, 1]. The negative control is `mean_baseline`, except
  `control_baseline` for the `deltapert` metrics, as in Miller et al. For
  `interpolated_duplicate` this share is the DRF. It keeps its sign, so a positive
  control that loses to the negative control shows up as negative. The proposal's
  formula takes absolute values, which would hide that.
- **Gene panels.** `stack_genes` is STACK's 15,012-gene vocabulary intersected with
  the dataset's genes detected in at least 3 cells. `hvg2000` is 2,000 highly variable
  genes chosen from training and control cells only.

The Cell-Eval-style metrics are computed on mean profiles rather than through the
`cell-eval` package, because baselines and TabPFN predict profiles, not cells.

## Norman et al. (2019): Study 2 setup

```bash
dibs submit --cpus 4 --mem 6G --time 30m -- uv run python scripts/norman_baselines.py
```

All 105 singles and 80% of the 131 doubles are training conditions. The other 26
doubles (seed 0) are held out. The split and the per-cell halves are saved in
`splits/`, so later STACK runs score against exactly the same cells. Results go to
`results/norman_baselines/`: `scores.csv` holds one row per panel × model ×
perturbation × metric, and `summary.md` holds the medians. The additive baseline's
`mse` per double is the additive residual the proposal uses to pick out non-additive
pairs.

This run differs from Miller et al.'s Norman pipeline on purpose in two ways:

- **Every cell is kept.** Miller et al. cap each perturbation at the mean cell count
  (about 470) and control at 8,192 cells. With more cells, the DE p-values are smaller,
  so the interpolated duplicate leans further toward the raw duplicate.
- **The genes are STACK's vocabulary**, not 8,192 HVGs plus the perturbed genes. As a
  result, 17 of the 105 perturbed genes have no score for their own on-target change.
  They are BCORL1, C19orf26, C3orf72, CSRNP1, DUSP9, FOXA1, FOXF1, FOXL2, HOXA13,
  HOXB9, HOXC13, KIAA1804, KMT2A, RHOXF2, S1PR2, SLC6A9 and TP73.
