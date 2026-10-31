# Reproducibility protocol

## Surrogate comparison

- Development data: 400 FE samples.
- Independent test data: 80 fixed FE samples.
- Development split per seed: 256 fitting, 64 early-stopping validation, and 80 variance-calibration samples.
- Seeds: 2025, 2026, 2027, 2028, and 2029.
- Scaling: feature-wise Z-score standardization fitted only on the fitting subset. The same fitted statistics are applied to validation, calibration, and test data.
- DE-MMoE: five independently initialized mean-only MMoE members.
- Training loss: standardized group-balanced MSE, assigning one third to `Volu`, one third to `St`, and one third collectively to the 39 stress responses.
- Checkpoint: validation macro-RMSE over the 41 standardized outputs.
- Model uncertainty: variance of the five member means.
- Probability evaluation: member variance plus a task-wise residual-variance floor fitted on the independent calibration subset.

The fixed test set means that the five runs quantify sensitivity to development splitting and initialization; they are not independent test datasets.

### Fitting-time definition

Table 6 reports fixed-configuration final-fitting wall-clock time, not grid-search or model-development time. The timed region includes model construction, final fitting, early stopping where applicable, and best-checkpoint restoration. It excludes hyperparameter search, data loading and split preparation, test inference, MC-Dropout sampling, metric calculation, and serialization. Every member uses one CPU thread; the five DE-MMoE members run in five isolated one-thread processes in parallel.

## Sample-size sensitivity

- Development budgets: 25, 50, ..., 400 samples.
- Five seeds per budget.
- The same 80-point independent test set is used at every budget.
- Reported quantities: macro-average R2 and physical-scale RMSE across all 41 outputs, summarized by mean and sample standard deviation.

## Optimizer comparison

- Common surrogate: fixed seed-2025 five-member DE-MMoE trained with all 400 development samples.
- Algorithms: NSGA-II, synchronous-batch MOEA/D, and MOPSO-CD.
- Population, reference-direction, and swarm sizes: 200.
- Seeds: 2025--2034.
- Budgets: 2,400; 5,000; 10,000; 15,000; 20,000; 30,000; 40,000; 50,000; 60,000; 80,000; 100,000; and 120,000 surrogate evaluations.
- Engineering checks are aggregated into binary hard feasibility: `CV=0` when every check passes and `CV=1` otherwise. Equal-CV solutions are compared by the optimizer's objective-fitness rule.
- HV ideal/reference points in physical objective space: `(240, 7.5)` and `(350, 30)`.
- Each reported budget uses the current nondominated front at that budget, not a cumulative historical front.
- The independent FE test set is not used by this comparison.

## A-DSE settings

- Initial archive: first 80 development samples; 64 fitting and 16 early-stopping validation samples.
- Outer rounds: 30.
- Representative points: minimum-volume, normalized-ideal-nearest, and minimum-`St` solutions.
- Gate: maximum of the two objective-wise RMS relative errors over the three representatives, strictly below 1%.
- AL batch: three current feasible-Pareto candidates.
- Acquisition: dynamically weighted geometric mean of the two raw model standard deviations.
- Exact-design duplicate tolerance: `1e-8` normalized RMS distance.
- Batch diversity distance: 0.025 normalized RMS distance.
- GradientSHAP: mean absolute attribution, per-objective maximum normalization, and equal-objective averaging.

The three representative FE points remain held out from surrogate fitting. Successful AL samples are added to the fitting archive regardless of the logged predicted-feasibility flag.

## Ablation-result status

The released logs do not all use the same NSGA-II search budget:

| Method | Online NSGA-II | Public status |
|---|---:|---|
| Full | 200 x 200 | formal |
| M1 | 100 x 100 | legacy online trajectory; a separate 200 x 200 PF replay is supplied |
| M2 | 200 x 200 | formal |
| M3 | 200 x 200 | formal |
| M4 | 100 x 100 | legacy |

The M1 replay changes only Pareto search on the stored surrogate/archive trajectory; it is not an online 200 x 200 rerun and does not reproduce a new AL trajectory. M1 and M4 therefore must not be described as strictly controlled 200 x 200 comparisons. Raw metadata are retained so this difference is auditable. See `Revise/adaptive_dse/public_results/ablation/ABLATION_PROTOCOL_NOTE.md`.

## Excluded components

Automatic geometry construction, Rhino/ANSYS invocation, solver databases, and machine-specific paths are excluded. The FE response workbook is sufficient to rerun the surrogate comparison and optimizer comparison. Reproducing new A-DSE FE additions requires an external evaluator returning the documented 41-output vector.
