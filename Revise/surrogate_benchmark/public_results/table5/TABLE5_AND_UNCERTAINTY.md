# Revised Table 5 and uncertainty comparison

## Table 5

| Model | R² | RMSE | Calibrated NLL | Time (min) |
| --- | --- | --- | --- | --- |
| GPR | 0.987515 ± 0.000401 | 0.035882 ± 0.000306 | -1.038882 ± 0.004672 | 42.511 ± 9.127 |
| XGB | 0.934781 ± 0.002795 | 0.130497 ± 0.001967 | — | 7.572 ± 1.684 |
| MLP | 0.983940 ± 0.000385 | 0.046190 ± 0.001616 | -0.862340 ± 0.014466 | 7.780 ± 0.736 |
| PCA-GPR | 0.986177 ± 0.000316 | 0.049567 ± 0.001517 | -0.904209 ± 0.014395 | 12.040 ± 2.948 |
| MT-MLP | 0.984555 ± 0.000331 | 0.054643 ± 0.001623 | -0.822806 ± 0.027078 | 3.350 ± 0.202 |
| MMoE | 0.984305 ± 0.000371 | 0.045118 ± 0.000662 | -0.850510 ± 0.007267 | 8.437 ± 5.314 |
| DE-MMoE | 0.986493 ± 0.000313 | 0.038864 ± 0.000633 | -0.982328 ± 0.015459 | 18.041 ± 4.836 |

Values are mean ± sample standard deviation over five development splits; the fixed 80-point test set is shared. Time is the recorded nine-configuration search workflow and is not manuscript-final because formal runs were executed concurrently.

## Predictive uncertainty quality

| Model | Raw NLL | Calibrated NLL | PICP95 | MPIW95 | Calibration error |
| --- | --- | --- | --- | --- | --- |
| GPR | 0.115622 ± 0.049719 | -1.038882 ± 0.004672 | 0.939634 ± 0.005752 | 0.377147 ± 0.014916 | 0.011890 ± 0.004434 |
| MLP | -0.644991 ± 0.067662 | -0.862340 ± 0.014466 | 0.940976 ± 0.005528 | 0.448199 ± 0.011224 | 0.012226 ± 0.002429 |
| PCA-GPR | -0.858169 ± 0.012675 | -0.904209 ± 0.014395 | 0.948841 ± 0.004276 | 0.407667 ± 0.009988 | 0.010290 ± 0.008662 |
| MT-MLP | -0.699439 ± 0.024810 | -0.822806 ± 0.027078 | 0.936829 ± 0.007328 | 0.442849 ± 0.014584 | 0.013979 ± 0.003538 |
| MMoE | — | -0.850510 ± 0.007267 | 0.945732 ± 0.004496 | 0.433958 ± 0.009476 | 0.013796 ± 0.004887 |
| DE-MMoE | 4.391483 ± 1.023141 | -0.982328 ± 0.015459 | 0.947683 ± 0.002464 | 0.396687 ± 0.004720 | 0.016204 ± 0.002681 |

## Uncertainty-error ranking on the two objectives

| Model | Uncertainty | Spearman | AUROC | Top-20% enrichment |
| --- | --- | --- | --- | --- |
| GPR | total predictive variance | 0.276292 ± 0.069580 | 0.676367 ± 0.063958 | 1.451749 ± 0.061383 |
| MLP | MC Dropout model variance | 0.310384 ± 0.096851 | 0.714258 ± 0.067861 | 1.469962 ± 0.246928 |
| PCA-GPR | reconstructed total variance | 0.378964 ± 0.069533 | 0.708008 ± 0.060041 | 1.306798 ± 0.140835 |
| MT-MLP | MC Dropout model variance | 0.346662 ± 0.133394 | 0.759180 ± 0.076036 | 1.569853 ± 0.178209 |
| DE-MMoE | ensemble model variance | 0.229334 ± 0.133829 | 0.648633 ± 0.056075 | 1.415828 ± 0.146536 |

The MMoE single member has no epistemic component; its ranking row uses conditional residual variance and must not be interpreted as model uncertainty. Cross-method ranking values describe each model's own error and do not establish active-learning superiority.
