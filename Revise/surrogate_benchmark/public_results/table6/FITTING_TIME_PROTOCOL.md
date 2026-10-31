# Unified fixed-configuration runtime benchmark

All entries are newly measured final-fitting wall-clock times after hyperparameters were frozen.
The timed region excludes hyperparameter search, data preparation, test inference, MC-Dropout sampling, metric calculation and result serialization.
Every individual model uses one numerical CPU thread. The five DE-MMoE members use five isolated one-thread CPU processes in parallel; process startup is included in its wall time.
Values are mean ± sample standard deviation over the five frozen development partitions.

| Model | Final fitting time (min) | Five-seed total (min) |
| --- | ---: | ---: |
| GPR | 3.951 ± 0.295 | 19.754 |
| XGB | 1.042 ± 0.134 | 5.210 |
| MLP | 2.386 ± 0.225 | 11.928 |
| PCA-GPR | 1.721 ± 0.068 | 8.604 |
| MT-MLP | 1.198 ± 0.139 | 5.989 |
| MMoE | 0.632 ± 0.124 | 3.160 |
| DE-MMoE | 0.743 ± 0.105 | 3.713 |
