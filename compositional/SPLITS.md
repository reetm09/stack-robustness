# Experiment 2 — how the data is split for each question

Status: **design only, under review.** No split files have been generated yet (step 2.7 in PLAN.md).

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
- **Output:** `splits/compositional/s_all_folds.csv` (double, fold).

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
**Output:** `splits/compositional/s_gene_seed{0..4}.csv` (condition, kind, split, category).

### S-order — order shortcut check (optional, Q3/Q5)
- Fine-tune STACK only on order A→B (alphabetically first gene as the prompt).
- Evaluate B→A on training doubles and on S-gene test doubles.
- If B→A is much worse, fine-tuning learned an order-specific shortcut.

## Additive vs non-additive pairs (Q6, used in every split)

**In short:** a pair is "non-additive" when its measured effect differs from the sum of its two
single effects by much more than measurement noise could explain.

**Definition** (fixed before any model result; it depends on the data only):
1. Use the pair's top-100 DE genes (A+B against control), within the `hvg2000` panel (scanpy top
   2,000 HVGs, chosen from training and control cells only).
2. Compute the residual share $s = \lVert r_{AB}\rVert^2 / \lVert \delta_{AB} \rVert^2$, where
   $r_{AB} = \delta_{AB} - (\delta_A + \delta_B)$.
3. Compute a noise floor: resample cells with replacement within A, B, A+B and control (200 draws),
   and recompute $\lVert r_{AB}\rVert^2$ each time. Separately, compute the same statistic for an
   "additive null": centre the A+B cells so their mean equals $\delta_A+\delta_B$, resample, and
   take the null mean and sd.
4. **Non-additive** if $s \ge 0.2$ and $z = (\lVert r\rVert^2 - \text{null mean}) / \text{null sd} \ge 3$.
5. Computed on the **duplicate (second) half** of the A+B cells, so the truth half used for scoring
   never chooses the pairs.

Study 4's list (9 test pairs, ranked by additive wMSE) is also reported, for comparison only.

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
