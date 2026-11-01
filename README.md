# A-DSE arch-dam dataset and code

This repository provides the finite-element dataset and the core code used for surrogate benchmarking, optimizer comparison, and A-DSE ablation studies.

## Contents

- `Revise/dataset/SurrogateDataset.xlsx`: 400 development samples and 80 independent test samples.
- `Revise/surrogate_benchmark/`: surrogate-model implementations and repeated-test results.
- `Revise/adaptive_dse/`: A-DSE, ablation, and optimizer-comparison code.
- `DATA_DICTIONARY.md`: variable definitions, units, bounds, and scaling.
- `REPRODUCIBILITY.md`: experimental settings and public-result locations.

The Rhino/ANSYS automation and solver project files are not included. The supplied FE dataset is sufficient for the surrogate and optimizer comparisons. Manuscript plotting workbooks and full A-DSE run logs are not distributed.

## Setup

```bash
python -m pip install -r requirements.txt
python verify_public_package.py
```

## Surrogate benchmark

```bash
python -m Revise.surrogate_benchmark.run_benchmark --models all --threads 1 --overwrite
python -m Revise.surrogate_benchmark.run_fixed_mmoe_runtime --ensemble-workers 5 --overwrite
python -m Revise.surrogate_benchmark.run_fixed_runtime --overwrite
python -m Revise.surrogate_benchmark.build_table6_results
```

Table 6 uses fixed-configuration fitting time. It excludes hyperparameter search, data preparation, inference, metric calculation, and file output. Seeds 2025--2029 use the same 80-point test set and different development-data partitions.

## Result files

- Table 6: `Revise/surrogate_benchmark/public_results/table6/table6_summary.csv`
- Table 7: `Revise/surrogate_benchmark/public_results/table6/table7_summary.csv`
- Sample-size study: `Revise/surrogate_benchmark/public_results/sample_size/`
- Optimizer comparison: `Revise/adaptive_dse/public_results/optimizer_comparison/`
- Ablation protocol: `Revise/adaptive_dse/ablation_protocol.csv`
- Pareto FE validation: `Revise/adaptive_dse/public_results/pareto_fe_validation/`

The ablation implementations and common settings are included, but manuscript-specific plotting tables, intermediate checkpoints, and per-round optimization logs are omitted.

Dataset SHA-256: `1f9aa8db86de10d3024856d906ef5324724cf1661039ab6b2bfad9bca58ff353`
