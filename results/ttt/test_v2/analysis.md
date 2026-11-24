# Study 4 analysis: test_v2

Primary metric: wmse on `stack_genes` (lower is better), diff-scored STACK, T = 1. Pairs: 26.

## Median wMSE

| prediction | all (26) | without pilot (23) | non-additive (9) |
|---|---|---|---|
| icl (diff) | 0.0168 | 0.0163 | 0.0376 |
| shared (diff) | 0.0128 | 0.0116 | 0.0335 |
| ttt (diff) | 0.0165 | 0.0143 | 0.0371 |
| ttt_random (diff) | 0.0177 | 0.0160 | 0.0371 |
| shared_ttt (diff) | 0.0108 | 0.0106 | 0.0314 |
| shared_ttt_random (diff) | 0.0138 | 0.0109 | 0.0355 |
| icl (raw) | 0.177 | 0.175 | 0.216 |
| shared (raw) | 0.0243 | 0.0205 | 0.0503 |
| ttt (raw) | 0.0408 | 0.0416 | 0.0441 |
| ttt_random (raw) | 0.0611 | 0.0595 | 0.0872 |
| shared_ttt (raw) | 0.0119 | 0.0119 | 0.0171 |
| shared_ttt_random (raw) | 0.0331 | 0.0239 | 0.0670 |
| technical_duplicate | 0.0024 | 0.0024 | 0.0024 |
| interpolated_duplicate | 0.0052 | 0.0056 | 0.0041 |
| additive | 0.0068 | 0.0064 | 0.0114 |
| query | 0.0155 | 0.0141 | 0.0347 |
| query_one | 0.0244 | 0.0234 | 0.0280 |
| mean_baseline | 0.0457 | 0.0416 | 0.104 |
| control_baseline | 0.0553 | 0.0503 | 0.117 |

## Paired differences in wMSE

Each row is B minus A per pair, so a positive value means A is better. The interval is a bootstrap 95% interval for the median (10,000 resamples of pairs).

| A vs B | pairs | median | 95% interval | pairs where A is better |
|---|---|---|---|---|
| shared_ttt vs shared (all) | 26 | 0.0009 | [0.0004, 0.0012] | 20 of 26 |
| shared_ttt vs shared (without pilot) | 23 | 0.0008 | [0.0003, 0.0010] | 17 of 23 |
| shared_ttt vs shared (non-additive) | 9 | 0.0014 | [-0.0012, 0.0058] | 7 of 9 |
| shared_ttt vs shared_ttt_random (all) | 26 | 0.0008 | [0.0001, 0.0013] | 18 of 26 |
| shared_ttt vs shared_ttt_random (without pilot) | 23 | 0.0009 | [0.0001, 0.0015] | 17 of 23 |
| shared_ttt vs shared_ttt_random (non-additive) | 9 | 0.0001 | [-0.0016, 0.0097] | 5 of 9 |
| ttt vs icl (all) | 26 | 0.0009 | [0.0001, 0.0014] | 19 of 26 |
| ttt vs icl (without pilot) | 23 | 0.0010 | [0.0001, 0.0014] | 18 of 23 |
| ttt vs icl (non-additive) | 9 | 0.0014 | [-0.0000, 0.0058] | 6 of 9 |
| ttt vs ttt_random (all) | 26 | 0.0001 | [-0.0002, 0.0005] | 15 of 26 |
| ttt vs ttt_random (without pilot) | 23 | 0.0002 | [-0.0001, 0.0005] | 14 of 23 |
| ttt vs ttt_random (non-additive) | 9 | 0.0000 | [-0.0011, 0.0060] | 5 of 9 |
| shared vs icl (all) | 26 | 0.0036 | [0.0011, 0.0072] | 21 of 26 |
| shared vs icl (without pilot) | 23 | 0.0032 | [0.0009, 0.0055] | 19 of 23 |
| shared vs icl (non-additive) | 9 | 0.0114 | [-0.0005, 0.0253] | 7 of 9 |
| icl vs query (all) | 26 | -0.0015 | [-0.0032, -0.0007] | 3 of 26 |
| icl vs query (without pilot) | 23 | -0.0012 | [-0.0029, -0.0004] | 3 of 23 |
| icl vs query (non-additive) | 9 | -0.0035 | [-0.0123, -0.0004] | 1 of 9 |
| icl vs additive (all) | 26 | -0.0115 | [-0.0170, -0.0085] | 2 of 26 |
| icl vs additive (without pilot) | 23 | -0.0100 | [-0.0153, -0.0062] | 2 of 23 |
| icl vs additive (non-additive) | 9 | -0.0261 | [-0.0674, -0.0062] | 1 of 9 |
| shared vs query (all) | 26 | 0.0019 | [-0.0006, 0.0032] | 17 of 26 |
| shared vs query (without pilot) | 23 | 0.0023 | [0.0000, 0.0039] | 16 of 23 |
| shared vs query (non-additive) | 9 | 0.0013 | [-0.0019, 0.0152] | 5 of 9 |
| shared vs additive (all) | 26 | -0.0065 | [-0.0141, -0.0043] | 3 of 26 |
| shared vs additive (without pilot) | 23 | -0.0061 | [-0.0128, -0.0035] | 3 of 23 |
| shared vs additive (non-additive) | 9 | -0.0185 | [-0.0334, 0.0040] | 2 of 9 |
| ttt vs query (all) | 26 | -0.0007 | [-0.0031, 0.0001] | 9 of 26 |
| ttt vs query (without pilot) | 23 | -0.0005 | [-0.0026, 0.0001] | 9 of 23 |
| ttt vs query (non-additive) | 9 | -0.0035 | [-0.0170, -0.0005] | 1 of 9 |
| ttt vs additive (all) | 26 | -0.0091 | [-0.0163, -0.0071] | 2 of 26 |
| ttt vs additive (without pilot) | 23 | -0.0086 | [-0.0162, -0.0061] | 2 of 23 |
| ttt vs additive (non-additive) | 9 | -0.0256 | [-0.0660, -0.0004] | 1 of 9 |
| ttt_random vs query (all) | 26 | -0.0005 | [-0.0036, 0.0001] | 9 of 26 |
| ttt_random vs query (without pilot) | 23 | -0.0003 | [-0.0009, 0.0002] | 9 of 23 |
| ttt_random vs query (non-additive) | 9 | -0.0041 | [-0.0125, -0.0003] | 1 of 9 |
| ttt_random vs additive (all) | 26 | -0.0108 | [-0.0166, -0.0075] | 2 of 26 |
| ttt_random vs additive (without pilot) | 23 | -0.0094 | [-0.0166, -0.0065] | 2 of 23 |
| ttt_random vs additive (non-additive) | 9 | -0.0257 | [-0.0664, -0.0065] | 1 of 9 |
| shared_ttt vs query (all) | 26 | 0.0030 | [0.0002, 0.0054] | 18 of 26 |
| shared_ttt vs query (without pilot) | 23 | 0.0033 | [0.0002, 0.0054] | 16 of 23 |
| shared_ttt vs query (non-additive) | 9 | 0.0071 | [-0.0023, 0.0268] | 6 of 9 |
| shared_ttt vs additive (all) | 26 | -0.0057 | [-0.0131, -0.0017] | 4 of 26 |
| shared_ttt vs additive (without pilot) | 23 | -0.0056 | [-0.0118, -0.0017] | 4 of 23 |
| shared_ttt vs additive (non-additive) | 9 | -0.0183 | [-0.0257, 0.0040] | 2 of 9 |
| shared_ttt_random vs query (all) | 26 | 0.0013 | [-0.0001, 0.0034] | 17 of 26 |
| shared_ttt_random vs query (without pilot) | 23 | 0.0014 | [-0.0000, 0.0034] | 15 of 23 |
| shared_ttt_random vs query (non-additive) | 9 | 0.0038 | [-0.0009, 0.0111] | 6 of 9 |
| shared_ttt_random vs additive (all) | 26 | -0.0068 | [-0.0136, -0.0043] | 3 of 26 |
| shared_ttt_random vs additive (without pilot) | 23 | -0.0057 | [-0.0118, -0.0043] | 3 of 23 |
| shared_ttt_random vs additive (non-additive) | 9 | -0.0236 | [-0.0375, 0.0042] | 2 of 9 |

## TTT gain against the additive baseline's error (Spearman, all pairs)

- shared_ttt gain over shared: rho = 0.18 (p = 0.37, 26 pairs)
- ttt gain over icl: rho = 0.22 (p = 0.28, 26 pairs)

## T = 5 against T = 1 (median wMSE, all pairs)

| arm | T = 1 | T = 5 |
|---|---|---|
| icl | 0.0168 | 0.0162 |
| shared | 0.0128 | 0.0136 |
| ttt | 0.0165 | 0.0168 |
| shared_ttt | 0.0108 | 0.0110 |

## Prompt use (exploratory, T = 1)

Replacing the control prompt with A's cells shifts the prediction. For each pair and prompt order: the correlation of that shift with A's own effect on A's top 200 genes (r), and the shift's size relative to that effect.

| arm | pair-orders | median r | r above +0.3 | r below -0.3 | median relative size |
|---|---|---|---|---|---|
| icl | 52 | -0.24 | 1 | 19 | 0.15 |
| shared | 52 | +0.57 | 40 | 3 | 0.26 |
| ttt | 52 | +0.03 | 6 | 15 | 0.16 |
| ttt_random | 52 | -0.01 | 3 | 14 | 0.16 |
| shared_ttt | 52 | +0.60 | 39 | 1 | 0.26 |
| shared_ttt_random | 52 | +0.54 | 40 | 3 | 0.22 |

## Secondary metrics (median over all pairs, diff-scored, T = 1)

Panel `hvg2000`:

| metric | icl | shared | ttt | ttt_random | shared_ttt | shared_ttt_random | query | additive |
|---|---|---|---|---|---|---|---|---|
| wmse | 0.0529 | 0.0369 | 0.0517 | 0.0568 | 0.0290 | 0.0369 | 0.0448 | 0.0153 |
| pearson_deltactrl | 0.848 | 0.871 | 0.854 | 0.849 | 0.894 | 0.872 | 0.896 | 0.910 |
| pearson_deltapert | 0.672 | 0.737 | 0.682 | 0.684 | 0.776 | 0.752 | 0.723 | 0.853 |
| mse | 0.0028 | 0.0022 | 0.0026 | 0.0028 | 0.0020 | 0.0023 | 0.0021 | 0.0018 |

Panel `stack_genes`:

| metric | icl | shared | ttt | ttt_random | shared_ttt | shared_ttt_random | query | additive |
|---|---|---|---|---|---|---|---|---|
| wmse | 0.0168 | 0.0128 | 0.0165 | 0.0177 | 0.0108 | 0.0138 | 0.0155 | 0.0068 |
| pearson_deltactrl | 0.753 | 0.800 | 0.770 | 0.765 | 0.798 | 0.798 | 0.807 | 0.828 |
| pearson_deltapert | 0.454 | 0.635 | 0.509 | 0.486 | 0.657 | 0.622 | 0.575 | 0.699 |
| mse | 0.0014 | 0.0012 | 0.0013 | 0.0014 | 0.0010 | 0.0012 | 0.0012 | 0.0011 |

