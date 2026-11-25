# Norman val doubles: ICL, shared adapter, TTT

21 doubles, 84 training doubles. Settings in args.json.

## Panel `stack_genes`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_ttt_random_T1_diff | shared_ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.697 | 0.848 | 0.862 | 0.849 | 0.823 | 0.832 | 0.562 |
| spearman_deltactrl_de | nan | 0.760 | 0.879 | 0.878 | 0.877 | 0.866 | 0.859 | 0.618 |
| de_pr_auc | 0.042 | 0.573 | 0.700 | 0.726 | 0.673 | 0.660 | 0.661 | 0.355 |
| direction_agreement | 0.000 | 0.920 | 0.993 | 0.989 | 0.978 | 0.984 | 0.982 | 0.875 |
| top_overlap | 0.000 | 0.200 | 0.550 | 0.650 | 0.650 | 0.500 | 0.550 | 0.300 |
| mse | 0.004 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 |
| mae | 0.026 | 0.020 | 0.017 | 0.017 | 0.017 | 0.018 | 0.018 | 0.025 |
| wmse | 0.067 | 0.046 | 0.009 | 0.002 | 0.004 | 0.022 | 0.016 | 0.033 |
| pearson_deltapert | -0.254 | nan | 0.718 | 0.763 | 0.722 | 0.520 | 0.590 | 0.266 |
| pearson_deltactrl_top | nan | 0.684 | 0.960 | 0.985 | 0.964 | 0.958 | 0.948 | 0.781 |
| pearson_deltapert_top | -0.528 | nan | 0.942 | 0.975 | 0.936 | 0.845 | 0.886 | 0.559 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.850 | 0.900 | 0.850 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_ttt_random_T1_diff | shared_ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.421 | 0.489 | 0.387 | 0.303 | 0.303 | -0.391 |
| spearman_deltactrl_de | nan | 0.000 | 0.412 | 0.453 | 0.355 | 0.345 | 0.321 | -0.477 |
| de_pr_auc | -1.000 | 0.000 | 0.309 | 0.355 | 0.214 | 0.260 | 0.250 | -0.415 |
| direction_agreement | -1.000 | 0.000 | 0.929 | 0.810 | 0.721 | 0.797 | 0.791 | -0.905 |
| top_overlap | -0.250 | 0.000 | 0.357 | 0.500 | 0.500 | 0.300 | 0.286 | 0.067 |
| mse | -0.579 | 0.000 | 0.450 | 0.518 | 0.514 | 0.258 | 0.337 | -0.307 |
| mae | -0.256 | 0.000 | 0.124 | 0.116 | 0.130 | 0.055 | 0.087 | -0.271 |
| wmse | -0.391 | 0.000 | 0.844 | 0.921 | 0.854 | 0.533 | 0.642 | 0.241 |
| pearson_deltapert | 0.000 | nan | 0.797 | 0.828 | 0.777 | 0.544 | 0.665 | 0.411 |
| pearson_deltactrl_top | nan | 0.000 | 0.873 | 0.950 | 0.867 | 0.861 | 0.816 | 0.143 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.956 | 0.834 | 0.921 | 0.711 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.714 | 0.286 |

## Panel `hvg2000`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_ttt_random_T1_diff | shared_ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.630 | 0.917 | 0.947 | 0.902 | 0.901 | 0.899 | 0.620 |
| spearman_deltactrl_de | nan | 0.754 | 0.935 | 0.954 | 0.947 | 0.930 | 0.919 | 0.752 |
| de_pr_auc | 0.070 | 0.587 | 0.806 | 0.844 | 0.791 | 0.792 | 0.765 | 0.505 |
| direction_agreement | 0.000 | 0.867 | 0.990 | 0.996 | 0.967 | 0.990 | 0.978 | 0.813 |
| top_overlap | 0.000 | 0.300 | 0.650 | 0.700 | 0.700 | 0.600 | 0.600 | 0.350 |
| mse | 0.012 | 0.007 | 0.002 | 0.001 | 0.001 | 0.004 | 0.003 | 0.006 |
| mae | 0.034 | 0.028 | 0.018 | 0.013 | 0.015 | 0.021 | 0.020 | 0.028 |
| wmse | 0.206 | 0.150 | 0.017 | 0.003 | 0.006 | 0.055 | 0.039 | 0.086 |
| pearson_deltapert | -0.280 | nan | 0.846 | 0.910 | 0.869 | 0.679 | 0.750 | 0.373 |
| pearson_deltactrl_top | nan | 0.650 | 0.961 | 0.984 | 0.953 | 0.961 | 0.946 | 0.742 |
| pearson_deltapert_top | -0.412 | nan | 0.937 | 0.976 | 0.943 | 0.828 | 0.885 | 0.536 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.800 | 0.950 | 0.850 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | shared_ttt_random_T1_diff | shared_ttt_random_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.661 | 0.846 | 0.729 | 0.634 | 0.649 | 0.015 |
| spearman_deltactrl_de | nan | 0.000 | 0.636 | 0.830 | 0.750 | 0.651 | 0.607 | -0.163 |
| de_pr_auc | -1.000 | 0.000 | 0.552 | 0.660 | 0.486 | 0.505 | 0.458 | -0.153 |
| direction_agreement | -1.000 | 0.000 | 0.944 | 0.980 | 0.826 | 0.917 | 0.760 | -0.333 |
| top_overlap | -0.333 | 0.000 | 0.471 | 0.600 | 0.533 | 0.455 | 0.400 | 0.077 |
| mse | -0.492 | 0.000 | 0.666 | 0.818 | 0.751 | 0.461 | 0.534 | -0.024 |
| mae | -0.212 | 0.000 | 0.362 | 0.492 | 0.351 | 0.242 | 0.289 | -0.062 |
| wmse | -0.296 | 0.000 | 0.895 | 0.972 | 0.940 | 0.577 | 0.661 | 0.264 |
| pearson_deltapert | 0.000 | nan | 0.880 | 0.932 | 0.881 | 0.700 | 0.803 | 0.501 |
| pearson_deltactrl_top | nan | 0.000 | 0.889 | 0.955 | 0.889 | 0.878 | 0.831 | 0.174 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.946 | 0.837 | 0.903 | 0.661 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.692 | 0.857 | 0.500 |
