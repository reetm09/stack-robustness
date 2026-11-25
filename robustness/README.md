# Study 3: robustness (data quality)

How STACK, TabPFN-3.5 and simple baselines degrade when the prompt or query cells become
sparse, technically corrupted, contaminated, or come from another plate. The proposal's
"Robustness" section is the spec. This is the preliminary version on one dataset,
OpenProblems PBMC donor 1 (Arc's `Stack-DrugPBMC-Example`, 32,530 cells, 14 drugs + DMSO).

## Run

From the repo root, in an environment with `stack_robustness` (`..`) and `stack_ttt` (`../ttt`):

```bash
pip install -e robustness                     # or: conda env from ../environment.yml
python scripts/fetch_hf.py                    # STACK, TabPFN-3.5, the PBMC file (into .cache/huggingface)
python -m stack_quality.run                   # about 30 min on one 8 GB laptop GPU; resumes if stopped
python -m pytest robustness/tests             # CPU only
```

Then open `robustness/scripts/robustness_figure.ipynb` for the figure.

## Task

Proposal Eq. 1, per target = (drug, query cell type, plate):

- **prompt** drug-treated T/NK cells, **control prompt** DMSO T/NK cells, **query** DMSO B and
  myeloid cells, all from the target's plate (12 drugs are on one plate only);
- **truth** the target's drug-treated B or myeloid cells, first random half; the second half is
  the technical duplicate. Truth is never corrupted.
- 31 targets with at least 30 cells. B and myeloid cells are scored separately: each drug's
  B/myeloid mix differs from DMSO's, which would otherwise read as a drug effect.

Default sizes fill STACK's window exactly (T = 1: 333 prompt + 179 query cells per 512-cell
window): up to 666 prompt cells and 179 query cells per cell type (2 windows per call).

## Models

| model | what it is |
|---|---|
| `stack` | frozen Stack-Large-Aligned, T = 1; generated(drug prompt) − generated(DMSO prompt), same query and seed |
| `stack_raw` | generated(drug prompt) − observed query |
| `tabpfn`, `ridge`, `knn` | prompt effect → query effect in a 50-PC space (PCA of DMSO cells, 2,000 HVGs), leave-one-drug-out, one pseudobulk row per target; TabPFN one regressor per PC |
| `source_effect_copy` | the prompt's T/NK effect copied onto the query |
| baselines | control, leave-one-drug-out mean, technical and interpolated duplicate; `pca_ceiling` = truth projected on the 50 PCs |

All predictions are an effect added to the target group's clean DMSO profile and scored with the
shared `stack_robustness` metrics. Tabular models are scored on the hvg panel only.

## Conditions (`conditions.py`)

| arm | settings | draws |
|---|---|---|
| clean | defaults | 3 |
| prompt size | 16, 32, 64, 128 cells (drug and DMSO prompt) | 2 |
| query size | 16, 32, 64, 128 cells per cell type | 2 |
| thinning | ρ = 10, 25, 50% × prompt / query / both | 2 |
| ambient | α = 10, 20% × prompt / query (b = pooled count profile) | 2 |
| contamination | 10, 25, 50% of the drug prompt replaced by other drugs' T/NK cells | 2 |
| plate | prompt from the query's plate vs the other plate (Belinostat, Dabrafenib) | 5 |

Draws are paired: every condition starts from the same cell orderings, and smaller sets are
prefixes of larger ones, so differences between conditions are not sampling noise.

## Outputs

- `results/robustness/pbmc_donor1/`: `scores.csv.gz` (one row per condition × draw × target ×
  model × panel × metric, with the DRF-style `gap_closed`), `baselines.csv.gz`, `timing.csv`,
  `summary.md`, `config.json`, `figures/robustness.{png,pdf}` (from the notebook).
- `data/processed/robustness/pbmc_donor1/` (git-ignored): cached scoring contexts and raw
  predictions per condition and draw.

## Not covered here (and why)

- Donor and study holdouts, donor covariates (KPMP, Wells): one donor. Plate stands in for a
  covariate, on two drugs.
- Library-size scaling, doublets, gene amplification; STACK adaptation levels beyond zero-shot;
  scVI and the doubly robust estimator.
- Tabular-model randomness: one pseudobulk per training target and one truth/duplicate split.
  Repeated subsampled pseudobulks (as replicates and extra context rows, keeping a held-out
  drug's replicates out of its own context) and repeated splits are left for the full report.
  With 13 training drugs per held-out drug, the tabular models are data-starved here.
- Prompts below 333 cells are repeated inside STACK's window, so the prompt-size arm measures a
  small prompt repeated, not fewer prompt slots.
