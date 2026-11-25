# Exploratory Experiment 4 variants on the validation doubles

Median wmse on `stack_genes` (lower is better), diff-scored, T = 1. PROTOCOL.md, "Two exploratory variants", fixes how these are read.

| run | pairs | median |
|---|---|---|
| icl | 21 | 0.02501 |
| shared | 21 | 0.01668 |
| shared_ttt | 21 | 0.01544 |
| shared_ttt, ED only | 21 | 0.01542 |
| shared_ttt, kNN | 21 | 0.01685 |
| shared_ttt_random | 21 | 0.01625 |
| ttt | 21 | 0.02231 |
| ttt, ED only | 21 | 0.02235 |
| ttt, kNN | 21 | 0.02245 |
| ttt_random | 21 | 0.02240 |

## Paired differences

B minus A per pair, so positive means A is better; bootstrap 95% interval for the median (10,000 resamples); ties are differences within 1% of B's value.

| A vs B | pairs | median | 95% interval | median change in % of B's median | A better / tie / B better |
|---|---|---|---|---|---|
| shared_ttt, ED only vs shared_ttt | 21 | +0.00001 | [-0.00019, +0.00006] | -0.2% | 2 / 13 / 6 |
| shared_ttt, kNN vs shared_ttt | 21 | +0.00000 | [-0.00111, +0.00013] | +9.1% | 8 / 3 / 10 |
| shared_ttt, kNN vs shared_ttt_random | 21 | +0.00033 | [+0.00005, +0.00115] | +3.7% | 12 / 4 / 5 |
| shared_ttt vs shared_ttt_random | 21 | +0.00072 | [+0.00000, +0.00212] | -5.0% | 14 / 1 / 6 |
| shared_ttt, ED only vs shared | 21 | +0.00040 | [-0.00007, +0.00141] | -7.6% | 13 / 3 / 5 |
| shared_ttt vs shared | 21 | +0.00058 | [-0.00009, +0.00132] | -7.5% | 14 / 2 / 5 |
| ttt, ED only vs ttt | 21 | -0.00005 | [-0.00019, +0.00001] | +0.2% | 2 / 14 / 5 |
| ttt, kNN vs ttt | 21 | +0.00000 | [-0.00019, +0.00021] | +0.6% | 8 / 6 / 7 |
| ttt, kNN vs ttt_random | 21 | +0.00030 | [+0.00000, +0.00132] | +0.2% | 12 / 5 / 4 |
| ttt vs ttt_random | 21 | +0.00034 | [+0.00000, +0.00101] | -0.4% | 11 / 6 / 4 |
| ttt, ED only vs icl | 21 | +0.00150 | [+0.00040, +0.00185] | -10.6% | 17 / 1 / 3 |
| ttt vs icl | 21 | +0.00153 | [+0.00079, +0.00182] | -10.8% | 17 / 1 / 3 |

`knn_shared_ttt`: the nearest doubles include 85 of the 150 gene-sharing neighbours (57%), over 21 pairs.

`knn_ttt`: the nearest doubles include 85 of the 150 gene-sharing neighbours (57%), over 21 pairs.
