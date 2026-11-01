# Reproducibility settings

## Surrogate comparison

- Development set: 400 FE samples.
- Independent test set: 80 FE samples.
- Split for each seed: 256 training, 64 validation, and 80 variance-calibration samples.
- Seeds: 2025--2029.
- Scaling: feature-wise Z-score statistics fitted on the training subset only.
- DE-MMoE: five independently initialized MMoE members.
- Training loss: standardized group-balanced MSE. `Volu`, `St`, and the 39 stress responses each receive one third of the total weight.
- Checkpoint: validation macro-RMSE across the 41 standardized outputs.
- Model uncertainty: variance of the five member means.
- Total predictive variance: model variance plus the calibrated residual-variance floor.

Table 6 reports mean and sample standard deviation across the five seeds. Fitting time covers model construction, training, early stopping, and checkpoint restoration. Each model uses one CPU thread. The five DE-MMoE members run in five one-thread processes.

## Sample-size study

- Development-set sizes: 25, 50, ..., 400.
- Five seeds per size.
- Fixed 80-point test set.
- Metrics: macro-average R2 and physical-scale RMSE across 41 outputs.

## Optimizer comparison

- Algorithms: NSGA-II, MOEA/D, and MOPSO-CD.
- Population, reference-direction, and swarm size: 200.
- Seeds: 2025--2034.
- Evaluation budgets: 2,400; 5,000; 10,000; 15,000; 20,000; 30,000; 40,000; 50,000; 60,000; 80,000; 100,000; and 120,000.
- Hard feasibility: `CV=0` when all checks pass and `CV=1` otherwise.
- HV ideal and reference points: `(240, 7.5)` and `(350, 30)`.
- HV is calculated from the nondominated front at each budget.

## A-DSE

- Initial samples: 80, including 16 early-stopping validation samples.
- Design-space updates: 30.
- Representative solutions: minimum volume, nearest to the normalized ideal point, and minimum `St`.
- Accuracy gate: the larger objective-wise RMS relative error over the three representatives must be below 1%.
- Active-learning batch: three feasible Pareto candidates.
- Acquisition score: dynamically weighted geometric mean of the two model standard deviations.
- Duplicate tolerance: `1e-8` normalized RMS distance.
- Batch diversity distance: `0.025` normalized RMS distance.
- GradientSHAP: mean absolute attribution, maximum normalization for each objective, and equal-objective averaging.

The three representative FE solutions are excluded from surrogate fitting. Active-learning FE samples are added to the training archive.

## Ablation configurations

| Method | Change | NSGA-II |
|---|---|---:|
| Full | Complete A-DSE | 200 x 200 |
| M1 | Fixed design space | 200 x 200 |
| M2 | Uniform expansion | 200 x 200 |
| M3 | Random active learning | 200 x 200 |
| M4 | No accuracy trigger | 200 x 200 |

All five configurations use the same initial dataset, outer-iteration count, NSGA-II budget, active-learning batch size, and random seed. Only the mechanism named in the `Change` column is altered. The machine-readable settings are provided in `Revise/adaptive_dse/ablation_protocol.csv`.

Full per-round optimization logs, model checkpoints, Pareto archives, and manuscript plotting workbooks are not part of the public package.
