# A-DSE arch-dam design: data and code

This repository contains the FE dataset, surrogate models, optimization code, and numerical results used in the revised manuscript.

## Contents

- `Revise/dataset/SurrogateDataset.xlsx`: 400 development samples and 80 independent test samples;
- `Revise/surrogate_benchmark/`: surrogate-model implementations and evaluation code;
- `Revise/surrogate_benchmark/public_results/`: Table 5, uncertainty, sample-size, and independent-test results;
- `Revise/adaptive_dse/`: A-DSE, ablations, and NSGA-II/MOEA-D/MOPSO-CD comparison code;
- `Revise/adaptive_dse/public_results/`: optimizer, ablation, and Pareto FE-validation results;
- `DATA_DICTIONARY.md`: variables, units, bounds, and preprocessing;
- `REPRODUCIBILITY.md`: experimental settings.

Rhino/ANSYS automation, machine-specific paths, CAD files, solver databases, and solver working files are not included.

## Surrogate models and repeated results

| Table 5 model | Main implementation |
|---|---|
| GPR | `models_sklearn.py: IndependentGPR` |
| XGB | `models_sklearn.py: IndependentXGBoost` |
| MLP with MC Dropout | `models_torch.py: IndependentTaskMLP`, `predict_mc_dropout` |
| PCA-GPR | `models_sklearn.py: PCALatentGPR` |
| MT-MLP with MC Dropout | `models_torch.py: MultiTaskMLP`, `predict_mc_dropout` |
| MMoE | `models_torch.py: DeterministicMMoE` |
| DE-MMoE | `models_torch.py: DeterministicMMoE`; ensemble fitting and aggregation in `run_benchmark.py` |

Per-run Table 5 results are stored in:

```text
Revise/surrogate_benchmark/public_results/table5/table5_by_seed.csv
```

Each model has five results for seeds 2025--2029. The 80-point FE test set is fixed; the seeds change the development-data split and neural-network initialization. The corresponding mean and sample standard deviation are in `table5_summary.csv`.

Probability and uncertainty-ranking results are stored in the same directory. XGB has no NLL or interval result because the implemented baseline does not produce a predictive variance.

## Dataset

| Sheet | Size | Description |
|---|---:|---|
| `TrainX` | 400 x 16 | development designs |
| `TrainY` | 400 x 41 | development FE responses |
| `ValX` | 80 x 16 | independent test designs |
| `ValY` | 80 x 41 | independent test FE responses |

The 41 responses are dam volume, the safety objective, and 39 principal-stress responses. See `DATA_DICTIONARY.md` for details.

## Run

```text
python -m pip install -r requirements.txt
python verify_public_package.py
python -m Revise.surrogate_benchmark.run_benchmark --models all --threads 1 --overwrite
python -m Revise.surrogate_benchmark.build_table5_results
python Revise/adaptive_dse/main_optimizer_comparison.py --preflight
```

Formal training and optimizer repetitions take substantially longer than the package check. Published results are retained under the two `public_results` directories.

## Integrity

- Dataset SHA-256: `d03a5af15e3ac03dd018f6922578473a206eadfb09be40621104a944e4a35c41`
- Fixed seed-2025 DE-MMoE SHA-256: `a4388e11d8f6ce6b87aa5e5852202d99a58e44e0066272b42674ee1bc09632af`
