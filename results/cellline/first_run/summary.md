# Study 1 first run (HEK293T → HCT116), 2026-11-24

The "First run" of `cellline/README.md`, on branch `stephen` at `0ae43d9` with Andrew's code from
`origin/andrew` at `f483002` on `PYTHONPATH`. Pipeline check only: about 15–25 cells per gene is
enough to test the pipeline, memory and runtime, not accuracy.

## Settings

Exactly the README's commands, with `--data hf://datasets/Xaira-Therapeutics/X-Atlas-Orion/data`
(streamed) and the split in `cellline/splits/xatlas_orion/`:

```
--train-genes 50 --val-genes 10 --max-cells-per-gene 256 --max-controls 2000
--batches-per-line 10 --steps 50          # all other options at their defaults
```

## Step 1: plan (`plan.json`, `step1_plan.log`)

| | |
|---|---|
| batch files | 10 HEK293T + 10 HCT116 |
| genes | 50 train, 10 val (none dropped for lack of cells) |
| perturbed cells | HEK293T 862, HCT116 1,482 |
| control cells | 2,000 train, 2,000 val |
| genes in STACK's vocabulary | 14,700 |

## Step 2: 50-step run (`run.json`, `val.json`, `history.jsonl`, `step2_run.log`)

| | before (zero-shot) | after 50 steps |
|---|---|---|
| validation loss | 10.4926 | **8.3901** |
| energy distance (`mmd_loss`) | 8.928 | 7.119 |
| f_CLS (`cls_loss`) | 0.765 | 0.507 |
| reconstruction (`recon_loss`) | 0.800 | 0.764 |
| sliced Wasserstein (`sw_loss`; weight `null` in run.json) | 0.074 | 0.077 |

- Validation loss fell for all 10 validation genes. PAIP2B is by far the hardest (29.86 → 24.82), then PTPRB (17.06 → 13.43).
- Training: **11.4 s**, peak GPU memory **1.53 GB** (one RTX 5000 Ada, 32 GB).
- Whole job: 5 min 48 s wall clock, almost all of it streaming the 20 batch files from Hugging Face; peak RAM 11.2 GB.
- No errors and no warnings.

## Environment (`versions.txt`)

Python 3.13.18, scanpy 1.13.0, scikit-learn 1.9.1, torch 2.14.1+cu126, arc-stack 0.1.3,
pandas 3.1.1, anndata 0.13.4, numpy 2.5.5. NVIDIA driver 535 (CUDA 12.2) runs torch's CUDA 12.6
build through minor-version compatibility. torch 2.15 (Andrew's lock) needs CUDA 13, which this
driver does not support.

## Notes

- `run.json`'s `base_checkpoint` was edited to the repo-relative path; the run used the same file.
- `adapter.pt` (2.8 MB) is not committed; it is a 50-step pipeline-check adapter.
- For full runs, a local copy of the release's `data/` folder (about 130 GB) will avoid re-streaming.
