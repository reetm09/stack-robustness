# Norman val doubles: ICL, shared adapter, TTT

21 doubles, 84 training doubles. Settings in args.json.

## Panel `stack_genes`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_T1_diff | ttt_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.697 | 0.848 | 0.862 | 0.849 | 0.823 | 0.778 | 0.449 |
| spearman_deltactrl_de | nan | 0.760 | 0.879 | 0.878 | 0.877 | 0.866 | 0.837 | 0.596 |
| de_pr_auc | 0.042 | 0.573 | 0.700 | 0.726 | 0.673 | 0.660 | 0.622 | 0.334 |
| direction_agreement | 0.000 | 0.920 | 0.993 | 0.989 | 0.978 | 0.984 | 0.970 | 0.891 |
| top_overlap | 0.000 | 0.200 | 0.550 | 0.650 | 0.650 | 0.500 | 0.450 | 0.050 |
| mse | 0.004 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 | 0.008 |
| mae | 0.026 | 0.020 | 0.017 | 0.017 | 0.017 | 0.018 | 0.019 | 0.036 |
| wmse | 0.067 | 0.046 | 0.009 | 0.002 | 0.004 | 0.022 | 0.022 | 0.045 |
| pearson_deltapert | -0.254 | nan | 0.718 | 0.763 | 0.722 | 0.520 | 0.505 | 0.245 |
| pearson_deltactrl_top | nan | 0.684 | 0.960 | 0.985 | 0.964 | 0.958 | 0.925 | 0.592 |
| pearson_deltapert_top | -0.528 | nan | 0.942 | 0.975 | 0.936 | 0.845 | 0.797 | 0.447 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.850 | 0.850 | 0.900 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_T1_diff | ttt_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.421 | 0.489 | 0.387 | 0.303 | 0.191 | -0.692 |
| spearman_deltactrl_de | nan | 0.000 | 0.412 | 0.453 | 0.355 | 0.345 | 0.166 | -0.625 |
| de_pr_auc | -1.000 | 0.000 | 0.309 | 0.355 | 0.214 | 0.260 | 0.116 | -0.422 |
| direction_agreement | -1.000 | 0.000 | 0.929 | 0.810 | 0.721 | 0.797 | 0.600 | -0.464 |
| top_overlap | -0.250 | 0.000 | 0.357 | 0.500 | 0.500 | 0.300 | 0.235 | -0.125 |
| mse | -0.579 | 0.000 | 0.450 | 0.518 | 0.514 | 0.258 | 0.258 | -1.000 |
| mae | -0.256 | 0.000 | 0.124 | 0.116 | 0.130 | 0.055 | 0.055 | -0.652 |
| wmse | -0.391 | 0.000 | 0.844 | 0.921 | 0.854 | 0.533 | 0.535 | -0.131 |
| pearson_deltapert | 0.000 | nan | 0.797 | 0.828 | 0.777 | 0.544 | 0.545 | 0.365 |
| pearson_deltactrl_top | nan | 0.000 | 0.873 | 0.950 | 0.867 | 0.861 | 0.729 | -0.088 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.956 | 0.834 | 0.793 | 0.629 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.333 | 0.429 | 0.750 |

## Panel `hvg2000`

Median score:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_T1_diff | ttt_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.630 | 0.917 | 0.947 | 0.902 | 0.901 | 0.850 | 0.497 |
| spearman_deltactrl_de | nan | 0.754 | 0.935 | 0.954 | 0.947 | 0.930 | 0.895 | 0.718 |
| de_pr_auc | 0.070 | 0.587 | 0.806 | 0.844 | 0.791 | 0.792 | 0.738 | 0.513 |
| direction_agreement | 0.000 | 0.867 | 0.990 | 0.996 | 0.967 | 0.990 | 0.959 | 0.877 |
| top_overlap | 0.000 | 0.300 | 0.650 | 0.700 | 0.700 | 0.600 | 0.500 | 0.250 |
| mse | 0.012 | 0.007 | 0.002 | 0.001 | 0.001 | 0.004 | 0.004 | 0.016 |
| mae | 0.034 | 0.028 | 0.018 | 0.013 | 0.015 | 0.021 | 0.022 | 0.040 |
| wmse | 0.206 | 0.150 | 0.017 | 0.003 | 0.006 | 0.055 | 0.052 | 0.083 |
| pearson_deltapert | -0.280 | nan | 0.846 | 0.910 | 0.869 | 0.679 | 0.649 | 0.324 |
| pearson_deltactrl_top | nan | 0.650 | 0.961 | 0.984 | 0.953 | 0.961 | 0.929 | 0.630 |
| pearson_deltapert_top | -0.412 | nan | 0.937 | 0.976 | 0.943 | 0.828 | 0.764 | 0.476 |
| nir | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.800 | 0.850 | 0.950 |

Median share of the gap closed:

| metric | control_baseline | mean_baseline | additive | technical_duplicate | interpolated_duplicate | query | ttt_T1_diff | ttt_T1_raw |
|---|---|---|---|---|---|---|---|---|
| pearson_deltactrl | nan | 0.000 | 0.661 | 0.846 | 0.729 | 0.634 | 0.445 | -0.319 |
| spearman_deltactrl_de | nan | 0.000 | 0.636 | 0.830 | 0.750 | 0.651 | 0.590 | -0.209 |
| de_pr_auc | -1.000 | 0.000 | 0.552 | 0.660 | 0.486 | 0.505 | 0.349 | -0.169 |
| direction_agreement | -1.000 | 0.000 | 0.944 | 0.980 | 0.826 | 0.917 | 0.667 | 0.038 |
| top_overlap | -0.333 | 0.000 | 0.471 | 0.600 | 0.533 | 0.455 | 0.231 | 0.000 |
| mse | -0.492 | 0.000 | 0.666 | 0.818 | 0.751 | 0.461 | 0.383 | -1.000 |
| mae | -0.212 | 0.000 | 0.362 | 0.492 | 0.351 | 0.242 | 0.186 | -0.350 |
| wmse | -0.296 | 0.000 | 0.895 | 0.972 | 0.940 | 0.577 | 0.595 | 0.087 |
| pearson_deltapert | 0.000 | nan | 0.880 | 0.932 | 0.881 | 0.700 | 0.667 | 0.454 |
| pearson_deltactrl_top | nan | 0.000 | 0.889 | 0.955 | 0.889 | 0.878 | 0.735 | 0.046 |
| pearson_deltapert_top | 0.000 | nan | 0.957 | 0.982 | 0.946 | 0.837 | 0.785 | 0.651 |
| nir | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.692 | 0.615 | 0.833 |
