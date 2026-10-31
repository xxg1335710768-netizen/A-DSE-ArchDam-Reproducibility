# Fixed-configuration fitting time

The timed region includes model construction, training, early stopping, and checkpoint restoration. It excludes hyperparameter search, data preparation, inference, metric calculation, and file output. Each model uses one CPU thread. The five DE-MMoE members run in five one-thread processes; process startup is included.

Values are mean ± sample standard deviation across seeds 2025--2029.

| Model | Fitting time (min) |
|---|---:|
| GPR | 3.951 ± 0.295 |
| XGB | 1.042 ± 0.134 |
| MLP | 2.386 ± 0.225 |
| PCA-GPR | 1.721 ± 0.068 |
| MT-MLP | 1.198 ± 0.139 |
| MMoE | 0.632 ± 0.124 |
| DE-MMoE | 0.743 ± 0.105 |
