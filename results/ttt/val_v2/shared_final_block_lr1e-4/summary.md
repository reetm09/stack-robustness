# Norman val doubles: ICL, shared adapter, TTT

21 doubles, 84 training doubles. Settings in args.json.

## Panel `stack_genes`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_T1_diff | shared_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.697 | 0.848 | 0.862 | 0.849 | 0.823 | 0.801 | 0.621 |
| spearman_deltactrl_de | nan | 0.760 | 0.879 | 0.878 | 0.877 | 0.866 | 0.837 | 0.670 |
| de_pr_auc | 0.042 | 0.573 | 0.700 | 0.726 | 0.673 | 0.660 | 0.650 | 0.455 |
| direction_agreement | 0.000 | 0.920 | 0.993 | 0.989 | 0.978 | 0.984 | 0.974 | 0.906 |
| top_overlap | 0.000 | 0.200 | 0.550 | 0.650 | 0.650 | 0.500 | 0.450 | 0.350 |
| mse | 0.004 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |
| mae | 0.026 | 0.020 | 0.017 | 0.017 | 0.017 | 0.018 | 0.018 | 0.023 |
| wmse | 0.067 | 0.046 | 0.009 | 0.002 | 0.004 | 0.022 | 0.020 | 0.031 |
| pearson_deltapert | -0.254 | nan | 0.718 | 0.763 | 0.722 | 0.520 | 0.499 | 0.257 |
| pearson_deltactrl_top | nan | 0.684 | 0.960 | 0.985 | 0.964 | 0.958 | 0.938 | 0.798 |
| pearson_deltapert_top | -0.528 | nan | 0.942 | 0.975 | 0.936 | 0.845 | 0.814 | 0.558 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.850 | 0.850 | 0.800 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_T1_diff | shared_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.421 | 0.489 | 0.387 | 0.303 | 0.235 | -0.249 |
| spearman_deltactrl_de | nan | 0.000 | 0.412 | 0.453 | 0.355 | 0.345 | 0.256 | -0.422 |
| de_pr_auc | -1.000 | 0.000 | 0.309 | 0.355 | 0.214 | 0.260 | 0.159 | -0.221 |
| direction_agreement | -1.000 | 0.000 | 0.929 | 0.810 | 0.721 | 0.797 | 0.689 | -0.348 |
| top_overlap | -0.250 | 0.000 | 0.357 | 0.500 | 0.500 | 0.300 | 0.294 | 0.059 |
| mse | -0.579 | 0.000 | 0.450 | 0.518 | 0.514 | 0.258 | 0.226 | -0.064 |
| mae | -0.256 | 0.000 | 0.124 | 0.116 | 0.130 | 0.055 | 0.063 | -0.163 |
| wmse | -0.391 | 0.000 | 0.844 | 0.921 | 0.854 | 0.533 | 0.562 | 0.247 |
| pearson_deltapert | 0.000 | nan | 0.797 | 0.828 | 0.777 | 0.544 | 0.572 | 0.431 |
| pearson_deltactrl_top | nan | 0.000 | 0.873 | 0.950 | 0.867 | 0.861 | 0.799 | 0.217 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.956 | 0.834 | 0.854 | 0.724 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.474 | 0.250 |

## Panel `hvg2000`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_T1_diff | shared_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.630 | 0.917 | 0.947 | 0.902 | 0.901 | 0.864 | 0.666 |
| spearman_deltactrl_de | nan | 0.754 | 0.935 | 0.954 | 0.947 | 0.930 | 0.902 | 0.774 |
| de_pr_auc | 0.070 | 0.587 | 0.806 | 0.844 | 0.791 | 0.792 | 0.755 | 0.520 |
| direction_agreement | 0.000 | 0.867 | 0.990 | 0.996 | 0.967 | 0.990 | 0.971 | 0.852 |
| top_overlap | 0.000 | 0.300 | 0.650 | 0.700 | 0.700 | 0.600 | 0.550 | 0.400 |
| mse | 0.012 | 0.007 | 0.002 | 0.001 | 0.001 | 0.004 | 0.003 | 0.005 |
| mae | 0.034 | 0.028 | 0.018 | 0.013 | 0.015 | 0.021 | 0.021 | 0.029 |
| wmse | 0.206 | 0.150 | 0.017 | 0.003 | 0.006 | 0.055 | 0.049 | 0.091 |
| pearson_deltapert | -0.280 | nan | 0.846 | 0.910 | 0.869 | 0.679 | 0.640 | 0.361 |
| pearson_deltactrl_top | nan | 0.650 | 0.961 | 0.984 | 0.953 | 0.961 | 0.942 | 0.761 |
| pearson_deltapert_top | -0.412 | nan | 0.937 | 0.976 | 0.943 | 0.828 | 0.799 | 0.593 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.800 | 0.850 | 0.750 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_T1_diff | shared_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.661 | 0.846 | 0.729 | 0.634 | 0.536 | -0.001 |
| spearman_deltactrl_de | nan | 0.000 | 0.636 | 0.830 | 0.750 | 0.651 | 0.627 | 0.080 |
| de_pr_auc | -1.000 | 0.000 | 0.552 | 0.660 | 0.486 | 0.505 | 0.379 | -0.116 |
| direction_agreement | -1.000 | 0.000 | 0.944 | 0.980 | 0.826 | 0.917 | 0.778 | -0.111 |
| top_overlap | -0.333 | 0.000 | 0.471 | 0.600 | 0.533 | 0.455 | 0.412 | 0.200 |
| mse | -0.492 | 0.000 | 0.666 | 0.818 | 0.751 | 0.461 | 0.365 | 0.061 |
| mae | -0.212 | 0.000 | 0.362 | 0.492 | 0.351 | 0.242 | 0.210 | -0.113 |
| wmse | -0.296 | 0.000 | 0.895 | 0.972 | 0.940 | 0.577 | 0.640 | 0.278 |
| pearson_deltapert | 0.000 | nan | 0.880 | 0.932 | 0.881 | 0.700 | 0.708 | 0.525 |
| pearson_deltactrl_top | nan | 0.000 | 0.889 | 0.955 | 0.889 | 0.878 | 0.839 | 0.278 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.946 | 0.837 | 0.830 | 0.690 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.692 | 0.692 | 0.273 |
