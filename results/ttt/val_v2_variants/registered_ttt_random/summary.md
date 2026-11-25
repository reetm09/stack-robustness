# Norman val doubles: ICL, shared adapter, TTT

21 doubles, 84 training doubles. Settings in args.json.

## Panel `stack_genes`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_random_T1_diff | ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.697 | 0.848 | 0.862 | 0.849 | 0.823 | 0.782 | 0.318 |
| spearman_deltactrl_de | nan | 0.760 | 0.879 | 0.878 | 0.877 | 0.866 | 0.827 | 0.475 |
| de_pr_auc | 0.042 | 0.573 | 0.700 | 0.726 | 0.673 | 0.660 | 0.629 | 0.258 |
| direction_agreement | 0.000 | 0.920 | 0.993 | 0.989 | 0.978 | 0.984 | 0.970 | 0.810 |
| top_overlap | 0.000 | 0.200 | 0.550 | 0.650 | 0.650 | 0.500 | 0.450 | 0.000 |
| mse | 0.004 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.009 |
| mae | 0.026 | 0.020 | 0.017 | 0.017 | 0.017 | 0.018 | 0.019 | 0.038 |
| wmse | 0.067 | 0.046 | 0.009 | 0.002 | 0.004 | 0.022 | 0.022 | 0.064 |
| pearson_deltapert | -0.254 | nan | 0.718 | 0.763 | 0.722 | 0.520 | 0.463 | 0.070 |
| pearson_deltactrl_top | nan | 0.684 | 0.960 | 0.985 | 0.964 | 0.958 | 0.933 | 0.327 |
| pearson_deltapert_top | -0.528 | nan | 0.942 | 0.975 | 0.936 | 0.845 | 0.764 | 0.217 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.850 | 0.850 | 0.600 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_random_T1_diff | ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.421 | 0.489 | 0.387 | 0.303 | 0.205 | -1.000 |
| spearman_deltactrl_de | nan | 0.000 | 0.412 | 0.453 | 0.355 | 0.345 | 0.144 | -1.000 |
| de_pr_auc | -1.000 | 0.000 | 0.309 | 0.355 | 0.214 | 0.260 | 0.157 | -0.582 |
| direction_agreement | -1.000 | 0.000 | 0.929 | 0.810 | 0.721 | 0.797 | 0.622 | -1.000 |
| top_overlap | -0.250 | 0.000 | 0.357 | 0.500 | 0.500 | 0.300 | 0.250 | -0.250 |
| mse | -0.579 | 0.000 | 0.450 | 0.518 | 0.514 | 0.258 | 0.190 | -1.000 |
| mae | -0.256 | 0.000 | 0.124 | 0.116 | 0.130 | 0.055 | 0.038 | -0.828 |
| wmse | -0.391 | 0.000 | 0.844 | 0.921 | 0.854 | 0.533 | 0.523 | -0.297 |
| pearson_deltapert | 0.000 | nan | 0.797 | 0.828 | 0.777 | 0.544 | 0.530 | 0.267 |
| pearson_deltactrl_top | nan | 0.000 | 0.873 | 0.950 | 0.867 | 0.861 | 0.784 | -0.713 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.956 | 0.834 | 0.786 | 0.396 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.421 | 0.143 |

## Panel `hvg2000`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_random_T1_diff | ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.630 | 0.917 | 0.947 | 0.902 | 0.901 | 0.859 | 0.319 |
| spearman_deltactrl_de | nan | 0.754 | 0.935 | 0.954 | 0.947 | 0.930 | 0.900 | 0.567 |
| de_pr_auc | 0.070 | 0.587 | 0.806 | 0.844 | 0.791 | 0.792 | 0.737 | 0.378 |
| direction_agreement | 0.000 | 0.867 | 0.990 | 0.996 | 0.967 | 0.990 | 0.968 | 0.753 |
| top_overlap | 0.000 | 0.300 | 0.650 | 0.700 | 0.700 | 0.600 | 0.500 | 0.200 |
| mse | 0.012 | 0.007 | 0.002 | 0.001 | 0.001 | 0.004 | 0.004 | 0.021 |
| mae | 0.034 | 0.028 | 0.018 | 0.013 | 0.015 | 0.021 | 0.021 | 0.047 |
| wmse | 0.206 | 0.150 | 0.017 | 0.003 | 0.006 | 0.055 | 0.055 | 0.168 |
| pearson_deltapert | -0.280 | nan | 0.846 | 0.910 | 0.869 | 0.679 | 0.599 | 0.074 |
| pearson_deltactrl_top | nan | 0.650 | 0.961 | 0.984 | 0.953 | 0.961 | 0.929 | 0.325 |
| pearson_deltapert_top | -0.412 | nan | 0.937 | 0.976 | 0.943 | 0.828 | 0.754 | 0.167 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.600 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_random_T1_diff | ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.661 | 0.846 | 0.729 | 0.634 | 0.496 | -0.965 |
| spearman_deltactrl_de | nan | 0.000 | 0.636 | 0.830 | 0.750 | 0.651 | 0.577 | -0.720 |
| de_pr_auc | -1.000 | 0.000 | 0.552 | 0.660 | 0.486 | 0.505 | 0.351 | -0.333 |
| direction_agreement | -1.000 | 0.000 | 0.944 | 0.980 | 0.826 | 0.917 | 0.750 | -1.000 |
| top_overlap | -0.333 | 0.000 | 0.471 | 0.600 | 0.533 | 0.455 | 0.250 | -0.077 |
| mse | -0.492 | 0.000 | 0.666 | 0.818 | 0.751 | 0.461 | 0.327 | -1.000 |
| mae | -0.212 | 0.000 | 0.362 | 0.492 | 0.351 | 0.242 | 0.194 | -0.676 |
| wmse | -0.296 | 0.000 | 0.895 | 0.972 | 0.940 | 0.577 | 0.597 | -0.175 |
| pearson_deltapert | 0.000 | nan | 0.880 | 0.932 | 0.881 | 0.700 | 0.667 | 0.305 |
| pearson_deltactrl_top | nan | 0.000 | 0.889 | 0.955 | 0.889 | 0.878 | 0.782 | -0.493 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.946 | 0.837 | 0.791 | 0.399 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.692 | 0.615 | 0.158 |
