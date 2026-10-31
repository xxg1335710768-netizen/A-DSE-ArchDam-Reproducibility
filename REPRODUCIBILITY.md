# Reproducibility protocol

## Surrogate comparison

- Development data: 400 FE samples.
- Independent test data: 80 FE samples, fixed across repetitions.
- Development split per seed: 256 training, 64 validation, and 80 calibration samples.
- Seeds: 2025, 2026, 2027, 2028, and 2029.
- Scaling: feature-wise training-only Z-score standardization.
- DE-MMoE: five independently initialized members.
- Seed-2025 selected architecture: two experts, expert width 128, task-tower width 32, Swish activation.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `5e-2`.
- Batch size: 32; maximum epochs: 500; patience: 30; gradient clip: 5.0.
- Loss: standardized group-balanced MSE, assigning one third to `Volu`, one third to `St`, and one third collectively to the 39 stress responses.
- Checkpoint: validation macro-RMSE over the 41 standardized outputs.
- Model uncertainty: member disagreement.
- Probability evaluation: member disagreement plus a task-wise additive residual-variance floor fitted on the calibration subset.

The fixed test set means that the five repetitions quantify sensitivity to development splitting and initialization; they are not five independent test sets.

## A-DSE settings

- Initial archive: first 80 development samples; 64 fitting and 16 early-stopping validation samples.
- Outer rounds: 30.
- NSGA-II population/generations: 200/200.
- Representative points: minimum-volume, normalized-ideal-nearest, and minimum-`St` solutions.
- Gate: maximum of the two objective-wise RMS relative errors over the three representatives, strictly below 1%.
- AL batch: three current feasible-Pareto candidates.
- Acquisition: dynamically weighted geometric mean of the two raw model standard deviations.
- Minimum objective weight: 0.20.
- Exact-design duplicate tolerance: `1e-8` normalized RMS distance.
- Batch diversity distance: 0.025 normalized RMS distance.
- Representative reuse distance: 0.01 normalized RMS distance.
- GradientSHAP: 30 baselines, 64 path samples, batch size 32; mean absolute attribution, per-objective maximum normalization, and equal objective averaging.
- A-DSE update: EMA coefficient 0.40; expansion factor decreases linearly from 0.15 to 0.03; reference-span ratio 0.10.

The three representative FE points remain held out from surrogate fitting. Successful AL samples are added to the training archive regardless of their logged predicted-feasibility flag.

## Ablations

| Method | Controlled change from Full |
|---|---|
| Full | Importance-weighted A-DSE + uncertainty AL + accuracy trigger |
| M1 | Fix the initial search bounds; retain uncertainty AL and gate |
| M2 | Remove only the `(1 + I_d)` importance gain from bound expansion |
| M3 | Replace uncertainty ranking by seeded random ranking on the same candidate pool |
| M4 | Add one three-point uncertainty batch per round and do not block the bound update with the gate |

All other settings are held fixed. `Revise/adaptive_dse/ablation_settings.py` records these definitions directly.

## Optimizer comparison

- Common surrogate: fixed seed-2025 five-member DE-MMoE.
- Algorithms: NSGA-II, synchronous batch MOEA/D, and MOPSO-CD.
- Population/swarm/reference directions: 200.
- Seeds: 2025--2034.
- Budgets: 2,400; 5,000; 10,000; 15,000; 20,000; 30,000; 40,000; 50,000; 60,000; 80,000; 100,000; and 120,000 surrogate evaluations.
- Each algorithm-seed pair follows one continuous trajectory.
- Engineering checks are aggregated into binary hard feasibility: `CV=0` if all checks pass and `CV=1` otherwise. Equal-CV solutions are compared by each optimizer's objective-fitness rule.
- HV ideal/reference points in physical objective space: `(240, 7.5)` and `(350, 30)`.
- The independent FE test set is not used by the optimizer comparison.

## Public results

- `surrogate_benchmark/public_results/table5`: five-seed predictive, probability, uncertainty-utility, and runtime summaries;
- `surrogate_benchmark/public_results/sample_size`: sample-size learning-curve data;
- `surrogate_benchmark/public_results/independent_test`: FE versus DE-MMoE prediction export;
- `adaptive_dse/public_results/optimizer_comparison`: formal three-algorithm results;
- `adaptive_dse/public_results/ablation`: Full and M1--M4 logs and Pareto snapshots;
- `adaptive_dse/public_results/pareto_fe_validation`: final Full-front prediction versus FE validation.

## Excluded project-specific components

The public package intentionally excludes automatic geometry construction, Rhino/ANSYS invocation, solver databases, and machine-specific paths. The FE response data used by the reported surrogate experiments is public in the workbook. The A-DSE method is exposed as callback-based algorithmic code; reproducing new FE additions requires an external evaluator returning the documented 41-output vector.
