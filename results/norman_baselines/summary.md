# Norman et al. (2019) baselines, Study 2 setup (seed 0)

26 held-out doubles scored against their truth half; 210 training conditions (all singles plus 105 doubles); 11855 control cells.

## Panel `stack_genes` (11789 genes)

Median score over held-out doubles:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate |
|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.699 | 0.828 | 0.838 | 0.829 |
| spearman_deltactrl_de | nan | 0.811 | 0.899 | 0.910 | 0.899 |
| de_pr_auc | 0.040 | 0.520 | 0.686 | 0.703 | 0.654 |
| direction_agreement | 0.000 | 0.924 | 0.988 | 0.987 | 0.975 |
| top_overlap | 0.000 | 0.250 | 0.575 | 0.600 | 0.625 |
| mse | 0.003 | 0.002 | 0.001 | 0.001 | 0.001 |
| mae | 0.024 | 0.020 | 0.017 | 0.016 | 0.016 |
| wmse | 0.055 | 0.046 | 0.007 | 0.002 | 0.005 |
| pearson_deltapert | -0.179 | nan | 0.699 | 0.746 | 0.680 |
| pearson_deltactrl_top | nan | 0.640 | 0.948 | 0.980 | 0.960 |
| pearson_deltapert_top | -0.366 | nan | 0.911 | 0.969 | 0.937 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 |

Median share of the gap from each metric's negative control to a perfect score (the `interpolated_duplicate` column is Miller et al.'s dynamic range fraction):

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate |
|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.497 | 0.493 | 0.415 |
| spearman_deltactrl_de | nan | 0.000 | 0.421 | 0.480 | 0.411 |
| de_pr_auc | -1.000 | 0.000 | 0.321 | 0.329 | 0.254 |
| direction_agreement | -1.000 | 0.000 | 0.882 | 0.895 | 0.706 |
| top_overlap | -0.333 | 0.000 | 0.446 | 0.515 | 0.456 |
| mse | -0.656 | 0.000 | 0.389 | 0.493 | 0.421 |
| mae | -0.275 | 0.000 | 0.104 | 0.107 | 0.101 |
| wmse | -0.449 | 0.000 | 0.836 | 0.925 | 0.861 |
| pearson_deltapert | 0.000 | nan | 0.748 | 0.748 | 0.682 |
| pearson_deltactrl_top | nan | 0.000 | 0.855 | 0.929 | 0.823 |
| pearson_deltapert_top | 0.000 | nan | 0.924 | 0.980 | 0.960 |
| nir | -0.024 | 0.000 | 1.000 | 1.000 | 1.000 |

## Panel `hvg2000` (2000 genes)

Median score over held-out doubles:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate |
|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.657 | 0.910 | 0.939 | 0.915 |
| spearman_deltactrl_de | nan | 0.755 | 0.924 | 0.951 | 0.943 |
| de_pr_auc | 0.067 | 0.583 | 0.820 | 0.831 | 0.791 |
| direction_agreement | 0.000 | 0.866 | 0.987 | 0.995 | 0.976 |
| top_overlap | 0.000 | 0.250 | 0.625 | 0.650 | 0.675 |
| mse | 0.006 | 0.005 | 0.002 | 0.001 | 0.001 |
| mae | 0.031 | 0.024 | 0.018 | 0.013 | 0.016 |
| wmse | 0.163 | 0.139 | 0.015 | 0.004 | 0.009 |
| pearson_deltapert | -0.206 | nan | 0.853 | 0.913 | 0.859 |
| pearson_deltactrl_top | nan | 0.545 | 0.947 | 0.980 | 0.952 |
| pearson_deltapert_top | -0.262 | nan | 0.911 | 0.970 | 0.931 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 |

Median share of the gap from each metric's negative control to a perfect score (the `interpolated_duplicate` column is Miller et al.'s dynamic range fraction):

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate |
|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.719 | 0.832 | 0.736 |
| spearman_deltactrl_de | nan | 0.000 | 0.695 | 0.823 | 0.772 |
| de_pr_auc | -1.000 | 0.000 | 0.553 | 0.615 | 0.507 |
| direction_agreement | -1.000 | 0.000 | 0.917 | 0.981 | 0.778 |
| top_overlap | -0.333 | 0.000 | 0.425 | 0.500 | 0.500 |
| mse | -0.573 | 0.000 | 0.659 | 0.823 | 0.710 |
| mae | -0.233 | 0.000 | 0.307 | 0.443 | 0.295 |
| wmse | -0.279 | 0.000 | 0.869 | 0.961 | 0.941 |
| pearson_deltapert | 0.000 | nan | 0.835 | 0.925 | 0.881 |
| pearson_deltactrl_top | nan | 0.000 | 0.862 | 0.938 | 0.873 |
| pearson_deltapert_top | 0.000 | nan | 0.921 | 0.979 | 0.953 |
| nir | -0.081 | 0.000 | 1.000 | 1.000 | 1.000 |
