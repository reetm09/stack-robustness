# Experiment 2 — how the data is split for each question

Status: **files written (step 2.7).** `python compositional/scripts/make_splits.py` (from the repo root)
recreates them exactly; `tests/test_splits.py` checks the leakage rules on the saved files.

## In short

To test a prediction method fairly, some gene pairs must be kept hidden while the method learns.
It is then graded only on those hidden pairs. How we choose the hidden pairs depends on the
question:

- **Questions that involve no learning** (does frozen STACK combine genes? which way does it fail?
  does the order matter?) can use **all 131 pairs**. Nothing is trained, so nothing can be leaked.
- **Questions about learning** (does fine-tuning help?) need hidden pairs, and *which* pairs we hide
  matters:
  - It is easier to predict a pair whose genes the model has already seen in other pairs (for
    example, KLF1 is in 11 pairs).
  - It is harder when one or both genes never appeared in any training pair.

  So we deliberately hold out some genes and report results separately for "both genes seen",
  "one gene new" and "both genes new".
- **The team's existing split** (26 random pairs) is kept unchanged, so our numbers can be compared
  with everyone else's.

In every split, the single-gene measurements for every gene are always available: the task is to
combine them, not to guess an unmeasured gene.

## What the data looks like (drives the design)

- 131 doubles built from 73 genes. Every one of those genes also has a measured single.
- How many doubles each gene is in:

  | doubles per gene | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | genes | 20 | 16 | 5 | 11 | 7 | 5 | 3 | 2 | 1 | 2 | 1 |

  Hubs: PTPN12 (12), KLF1 (11), CEBPE (11), UBASH3B (10).
- The shared random test set (26 pairs) has 19 pairs with both genes seen in training doubles, 6
  with one gene seen, and 1 with neither (KIF18B+KIF2C). It cannot answer the "new genes" question.
- Truth / duplicate halves: every condition's cells are split 50/50 at random
  (`splits/norman2019_halves.csv.gz`, team convention). Models are scored against the **first**
  half. The **second** half is the technical duplicate, a noise ceiling: how well one half of
  the real cells predicts the other half.

## The splits

### S-shared — team reference (unchanged)
- `splits/norman2019_conditions.csv` (seed 0): train = 105 singles + 105 doubles; test = 26 doubles.
- Validation (`splits/norman2019_val_conditions.csv`): 21 of the 105 training doubles.
- **Used for:** comparing with the team's baseline table and Study 4.

### S-all — every pair, no training (Q1 composition, Q2 behaviour, Q3 order)
- **Evaluated:** all 131 doubles, in both prompt orders (262 pair-orders).
- **Fitted:** only the weights $(w_A, w_B)$ of the `weighted` arm. These are **cross-fitted**: the
  131 doubles are split into 5 folds, and each fold's weights are chosen on the other 4 folds. No
  pair helps choose its own weights.
- **Cells:** the prompt (A singles) and query (B singles) are disjoint from the target (A+B first
  half). Frozen STACK sees no double-perturbation cells at all.
- **Folds are balanced by additivity group** (each fold has 19–20 additive, 4–5 buffering/synergy and 2
  interaction pairs), so no fold's weights are fitted without non-additive pairs.
- **Output:** `splits/compositional/s_all_folds.csv` (condition, fold, group); fold sizes 27/26/26/26/26.

### S-gene — held-out genes, 5 seeds (Q4 fine-tuning, Q5 order consistency)
For each seed:
1. Draw a held-out gene set G, about 15% of the 73 genes (≈ 11).
2. **Test** = every double containing a gene in G:
   - **unseen 2/2**: both genes in G;
   - **unseen 1/2**: exactly one gene in G.
   Plus a random 15% of the remaining doubles, labelled **seen 2/2**.
3. **Train** = every other double, all 105 singles, and controls. No training double contains a
   gene in G.
4. **Validation** = the same construction applied inside the training doubles (a second
   held-out gene set plus random pairs). It is used for every choice: weights, ridge λ, kNN k, and
   the fine-tuning steps and learning rate.
5. **Checks:** at least 5 test pairs in each category, and at least 60% of doubles left for
   training. If unseen 2/2 is too rare, resample G (up to 1,000 tries, keeping the first draw that
   passes) and record the seed.

Results are reported per category, with the median over seeds and a paired bootstrap CI.

**Category is counted from the final training pairs, not from the draw.** A gene outside G can
also end up unseen, when its only pairs were among the random 15% (e.g. MEIS1 in seed 0). Those
pairs are labelled 1/2 or 0/2 accordingly, and the extra genes are listed as `also_unseen` in the
metadata.

**Output** (team format, so `build_contexts.py --conditions … --held test|val` works directly):
- `splits/compositional/s_gene_seed{k}_conditions.csv`: condition, kind, split (train / test),
  category. Train includes the validation pairs, for the final run after settings are fixed.
- `splits/compositional/s_gene_seed{k}_val_conditions.csv`: split = train / val; test pairs are
  left out entirely (as in the team's `norman2019_val_conditions.csv`).
- `splits/compositional/s_gene_meta.json`: per seed, the held genes, `also_unseen`, tries and counts.

**What came out** (test pairs per category; training doubles; validation pairs):

| seed | held-out genes G | seen 2/2 | unseen 1/2 | unseen 2/2 | train doubles | val (2/2 · 1/2 · 0/2) |
|---|---|---|---|---|---|---|
| 0 | FOXA3, FOXL2, IER5L, PLK4, POU3F2, PRDM1, RUNX1T1, STIL, TMSB4X, UBASH3A, UBASH3B | 14 | 22 | 9 | 86 | 5 · 19 · 9 |
| 1 | BPGM, C19orf26, CDKN1B, CEBPE, KIF18B, MAP2K3, PLK4, RHOXF2, S1PR2, SAMD1, TBX2 | 14 | 25 | 7 | 85 | 7 · 17 · 8 |
| 2 | BCL2L11, BPGM, CBFA2T3, CEBPE, COL2A1, ELMSAN1, IRF1, ISL2, KIF2C, MAP2K6, PRTG | 10 | 31 | 8 | 82 | 9 · 14 · 5 |
| 3 | BCL2L11, CEBPE, COL2A1, DUSP9, IRF1, ISL2, LHX1, MAPK1, S1PR2, SGK1, ZC3HAV1 | 11 | 32 | 6 | 82 | 8 · 16 · 7 |
| 4 | C3orf72, ETS2, IRF1, ISL2, MAPK1, OSR2, SGK1, SPI1, TBX3, UBASH3A, ZBTB1 | 13 | 31 | 5 | 82 | 8 · 17 · 7 |

Every seed passed within 1–3 draws; validation within 1–10. Non-additive test pairs per seed: 10, 14,
19, 17, 11, mostly in "unseen 1/2". Interaction pairs are 0–1 per category outside "unseen 1/2", too
few to report per category, so the interaction table pools categories.

### S-order — order shortcut check (optional, Q3/Q5)
- Fine-tune STACK only on order A→B (alphabetically first gene as the prompt).
- Evaluate B→A on training doubles and on S-gene test doubles.
- If B→A is much worse, fine-tuning learned an order-specific shortcut.

## Additive vs non-additive pairs (Q6, used in every split)

**In short:** before any model is tested, we check each real pair by putting the simple guess
"A's effect + B's effect" on a ruler:
- **0** = as good as repeating the experiment (the second half of the pair's cells);
- **1** = as bad as guessing "nothing changes".

We do the same for a stretched/shrunk sum (c₁·A + c₂·B, with weights per pair):
- **additive**: the plain sum lands near 0;
- **buffering/synergy**: the plain sum is far, but the stretched sum lands near 0;
- **interaction**: both are far.

**Agreed rule (decided before any model was run):**
- The 25% of pairs whose plain sum scores worst are **non-additive** (cut = 0.117, the 75th
  percentile over all 131 doubles).
- Of those, pairs whose stretched sum *still* scores above the same cut are **interaction**; the rest
  are **buffering/synergy**.
- **Result:** 98 additive, 23 buffering/synergy, 10 interaction.
  - The 10 interaction pairs: CEBPE+PTPN12, DUSP9+MAPK1, FEV+CBFA2T3, FOSB+PTPN12, FOSB+UBASH3B,
    MAP2K6+SPI1, MAPK1+PRTG, PTPN12+SNAI1, SET+CEBPE, ZBTB10+DLX2.
  - The shared 26 test pairs: 20 additive, 5 buffering/synergy, 1 interaction (MAP2K6+SPI1). That is
    too few for an interaction table on S-shared, which is another reason Q1–Q3 use S-all.
- **Saved as:** `splits/compositional/additivity_groups.csv`, written by
  `scripts/make_additivity_groups.py`. The same labels are used in every split.
- **Why one cut:** taking the top 25% of the stretched score separately put 16 pairs in a
  meaningless group ("plain sum fine, stretched sum off"). Most stretched scores are tiny, and the
  stretched fit is least squares while the grading uses wMSE.

**Technical** (definitions from the reference methods: metric calibration and the specific
epistasis model; `stack_compositional.additivity`):
- score = (wMSE(guess) − wMSE(duplicate)) / (wMSE(control) − wMSE(duplicate)), clipped at 0.
  Pairs where the duplicate is not better than control are dropped: 0 of 131 on `hvg2000`, and
  KIF18B+KIF2C on `stack_genes`.
- wMSE, truth and duplicate are exactly the shared scoring's (halves mode; DE t-test vs rest on the
  first half). On the 26 shared test pairs, the plain-sum score equals the one computed from
  `results/compositional/s_shared/scores.csv`.
- The stretched sum is a joint OLS over all 131 doubles: per-pair c₁, c₂ and a shared gene offset ε.
  It is fit by unweighted least squares, so under wMSE it is not guaranteed to beat the plain sum
  (e.g. PTPN12+OSR2).
- Panels: `hvg2000` (main) and `stack_genes`.
- Study 4's 9-pair list (largest additive wMSE) is reported alongside for comparison.

## Scoring modes (apply to every split)

**In short:** every split is graded twice:
- against a random half of each held-out pair's cells (team convention; the other half is the
  "repeat experiment" reference);
- against all of the pair's cells (less noisy, no repeat-experiment reference).

The pairs, the training data and the non-additive flags are identical in both modes. Only the
"right answer" changes. See PLAN.md, "Two scoring modes".

## Leakage rules (tested in step 2.7)
- A test double never appears in training or validation, within the same split.
- Under S-gene, no training or validation double contains a held-out gene from G.
- No setting (weights, λ, k, fine-tuning hyperparameters) is chosen using test doubles.
- The 2,000 HVGs are chosen from training conditions and control cells only, separately for each split.
