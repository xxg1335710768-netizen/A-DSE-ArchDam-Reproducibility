# A-DSE arch-dam reproducibility package

This repository contains the FE input-output dataset, surrogate-model code, optimization code, and numerical results used in the revised manuscript.

## Contents

- `Revise/dataset/SurrogateDataset.xlsx`: 400 development samples and 80 independent test samples;
- `Revise/surrogate_benchmark/`: GPR, XGB, MLP, PCA-GPR, MT-MLP, MMoE, and five-member DE-MMoE implementations;
- `Revise/surrogate_benchmark/public_results/table6/`: five-run Table 6 prediction and fitting-time results plus Table 7 interval results;
- `Revise/surrogate_benchmark/public_results/sample_size/`: 25--400 sample learning-curve results;
- `Revise/adaptive_dse/`: A-DSE and NSGA-II/MOEA-D/MOPSO-CD source code;
- `Revise/adaptive_dse/public_results/optimizer_comparison/`: ten-run optimizer comparison;
- `Revise/adaptive_dse/public_results/ablation/`: A-DSE and ablation logs, with protocol status stated in `ABLATION_PROTOCOL_NOTE.md`;
- `Revise/adaptive_dse/public_results/pareto_fe_validation/`: final A-DSE front prediction versus FE validation;
- `DATA_DICTIONARY.md`: variables, units, bounds, and training-only standardization;
- `REPRODUCIBILITY.md`: experimental definitions and limitations;
- `AUDIT_NOTES.md`: corrections made before the public release was updated.

Rhino/ANSYS automation, CAD files, solver databases, and machine-specific working files are not included.

## Manuscript Table 6

The canonical five-seed results are:

```text
Revise/surrogate_benchmark/public_results/table6/table6_by_seed.csv
Revise/surrogate_benchmark/public_results/table6/table6_summary.csv
```

`Fitting time` is the wall-clock time required to fit the frozen final configuration. Hyperparameter search, data preparation, test inference, MC-Dropout sampling, metric calculation, and serialization are excluded. Each individual model uses one CPU thread. The five DE-MMoE members use five isolated one-thread processes in parallel, including process-startup overhead. The exact timing records and protocol are retained beside Table 6.

The 80-point FE test set is fixed. Seeds 2025--2029 change the development-data split and neural-network initialization; they do not represent five independent test sets. Error bars are sample standard deviations across those five runs.

Table 7 PICP and MPIW results are in `table7_by_seed.csv` and `table7_summary.csv`. XGB is omitted because the released XGB baseline does not produce a predictive variance.

## Dataset

| Sheet | Shape | Description |
|---|---:|---|
| `TrainX` | 400 x 16 | development designs |
| `TrainY` | 400 x 41 | development FE responses |
| `ValX` | 80 x 16 | independent test designs |
| `ValY` | 80 x 41 | independent test FE responses |

See `DATA_DICTIONARY.md` before interpreting stress signs or response columns.

## Verification

```text
python -m pip install -r requirements.txt
python verify_public_package.py
python -m unittest Revise.surrogate_benchmark.test_protocol Revise.surrogate_benchmark.test_models_sklearn Revise.surrogate_benchmark.test_models_torch Revise.adaptive_dse.test_adaptive_dse Revise.adaptive_dse.test_optimizer_comparison
```

To rerun the surrogate benchmark and the comparable fitting-time measurement:

```text
python -m Revise.surrogate_benchmark.run_benchmark --models all --threads 1 --overwrite
python -m Revise.surrogate_benchmark.run_fixed_mmoe_runtime --ensemble-workers 5 --overwrite
python -m Revise.surrogate_benchmark.run_fixed_runtime --overwrite
python -m Revise.surrogate_benchmark.build_table6_results
```

Formal training takes substantially longer than package verification. A-DSE source code is released with evaluator callbacks, but producing new FE samples requires the excluded project-specific geometry and solver pipeline.

## Integrity

- Dataset SHA-256: `d03a5af15e3ac03dd018f6922578473a206eadfb09be40621104a944e4a35c41`
- Fixed seed-2025 DE-MMoE SHA-256: `a4388e11d8f6ce6b87aa5e5852202d99a58e44e0066272b42674ee1bc09632af`
